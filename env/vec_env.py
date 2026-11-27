"""여러 :class:`~env.sumo_env.SumoHighwayEnv`를 각각 별도 프로세스에서 돌리는 환경 묶음.

각 환경은 자기 프로세스에서 SUMO를 돌리고, 메인 프로세스와는 파이프로 명령과 결과를
주고받는다. 명령을 보낸 뒤 결과를 기다리지 않고 다른 환경에 명령을 보낼 수 있어서,
한 환경이 리셋하는 동안 다른 환경은 계속 스텝을 진행할 수 있다.

워커 프로세스는 torch를 불러오지 않는다. ``spawn`` 방식은 원래 자식 프로세스에서
메인 스크립트(train.py)를 다시 import하는데, 그러면 torch까지 함께 올라온다. 그래서
워커를 띄우는 동안에만 메인 모듈의 ``__file__``을 숨긴다.

Example:
    >>> envs = SubprocEnvs(2, env_kwargs, seed=0)
    >>> envs.send(0, "reset", 0)
    >>> ready = envs.wait([0])
    >>> obs = envs.recv(0)
    >>> envs.close()
"""
import multiprocessing as mp
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.connection import Connection, wait
from typing import Any

import numpy as np


def _worker(conn: Connection, env_kwargs: dict, seed: int) -> None:
    """워커 프로세스 본체다. ``conn``으로 받은 명령을 환경에 실행하고 결과를 돌려준다.

    명령은 ``(이름, 데이터)`` 튜플이고, 응답은 ``("ok", 결과)`` 또는
    ``("error", 트레이스백 문자열)``이다.

    - ``step``: 데이터는 행동이다. ``env.step`` 결과 5-튜플을 돌려준다.
    - ``reset``: 데이터는 전체 학습 스텝 수다. ``env.total_steps``에 넣고 리셋한 뒤
      초기 관측을 돌려준다.
    - ``get``: 데이터는 속성 이름이다. 그 값을 돌려준다.
    - ``set``: 데이터는 ``(속성 이름, 값)``이다.
    - ``call``: 데이터는 ``(메서드 이름, 인자 튜플)``이다. 메서드 반환값을 돌려준다.
    - ``close``: 환경을 닫고 끝낸다.

    Args:
        conn: 메인 프로세스와 연결된 파이프 끝.
        env_kwargs: :class:`~env.sumo_env.SumoHighwayEnv` 생성 인자.
        seed: 이 프로세스의 numpy 전역 난수 시드.
    """
    from env.sumo_env import SumoHighwayEnv

    np.random.seed(seed)
    env = None
    try:
        env = SumoHighwayEnv(**env_kwargs)
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                conn.send(("ok", env.step(data)))
            elif cmd == "reset":
                env.total_steps = int(data)
                conn.send(("ok", env.reset()[0]))
            elif cmd == "get":
                conn.send(("ok", getattr(env, data)))
            elif cmd == "set":
                setattr(env, data[0], data[1])
                conn.send(("ok", None))
            elif cmd == "call":
                conn.send(("ok", getattr(env, data[0])(*data[1])))
            elif cmd == "close":
                conn.send(("ok", None))
                break
            else:
                raise ValueError(f"알 수 없는 명령: {cmd}")
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        conn.close()


@contextmanager
def _main_file_hidden() -> Iterator[None]:
    """블록 안에서 ``__main__`` 모듈의 ``__file__``을 숨긴다.

    ``spawn`` 자식 프로세스가 메인 스크립트를 다시 import하지 않게 한다.
    """
    main = sys.modules["__main__"]
    path = getattr(main, "__file__", None)
    if path is not None:
        del main.__file__
    try:
        yield
    finally:
        if path is not None:
            main.__file__ = path


class SubprocEnvs:
    """환경 ``n_envs``개를 각각 별도 프로세스에서 돌리는 묶음이다.

    환경 ``i``에는 한 번에 명령 하나만 보낼 수 있다. :meth:`send` 뒤에는 :meth:`recv`로
    응답을 받아야 다음 명령을 보낼 수 있다. 환경 ``i``의 numpy 시드는 ``seed + i``다.

    Args:
        n_envs: 환경 개수.
        env_kwargs: 모든 환경에 공통인 :class:`~env.sumo_env.SumoHighwayEnv` 생성 인자.
        seed: 첫 환경의 시드.
    """

    def __init__(self, n_envs: int, env_kwargs: dict, seed: int = 0):
        ctx = mp.get_context("spawn")
        self.n_envs = n_envs
        self.conns: list[Connection] = []
        self.procs: list[mp.process.BaseProcess] = []
        with _main_file_hidden():
            for i in range(n_envs):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_worker, args=(child, env_kwargs, seed + i),
                                   daemon=True)
                proc.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(proc)
        self._index = {conn: i for i, conn in enumerate(self.conns)}
        self.observation_space = self.call(0, "get", "observation_space")
        self.action_space = self.call(0, "get", "action_space")

    def send(self, i: int, cmd: str, data: Any = None) -> None:
        """환경 ``i``에 명령 ``cmd``를 보내고 응답을 기다리지 않는다."""
        self.conns[i].send((cmd, data))

    def recv(self, i: int) -> Any:
        """환경 ``i``의 응답을 받아 결과를 반환한다. 응답이 올 때까지 기다린다.

        Raises:
            RuntimeError: 워커에서 예외가 났거나 워커가 응답 없이 끝났다.
        """
        try:
            status, payload = self.conns[i].recv()
        except EOFError as e:
            raise RuntimeError(f"환경 {i} 워커가 응답 없이 끝났다.") from e
        if status == "error":
            raise RuntimeError(f"환경 {i} 워커 오류:\n{payload}")
        return payload

    def call(self, i: int, cmd: str, data: Any = None) -> Any:
        """환경 ``i``에 명령을 보내고 응답을 받아 반환한다."""
        self.send(i, cmd, data)
        return self.recv(i)

    def set_all(self, name: str, value: Any) -> None:
        """모든 환경의 속성 ``name``을 ``value``로 바꾼다."""
        for i in range(self.n_envs):
            self.send(i, "set", (name, value))
        for i in range(self.n_envs):
            self.recv(i)

    def wait(self, indices: list[int]) -> list[int]:
        """``indices`` 중 응답이 도착한 환경 번호들을 반환한다.

        하나 이상 도착할 때까지 기다린다. ``indices``는 비어 있으면 안 된다.
        """
        ready = wait([self.conns[i] for i in indices])
        return sorted(self._index[conn] for conn in ready)

    def close(self) -> None:
        """모든 워커를 닫는다. 응답이 없는 워커는 강제로 끝낸다."""
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        for conn in self.conns:
            conn.close()
