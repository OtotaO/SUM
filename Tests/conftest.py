import asyncio
import inspect


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as asynchronous")


def pytest_pyfunc_call(pyfuncitem):
    test_func = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_func):
        asyncio.run(test_func(**pyfuncitem.funcargs))
        return True
    return None
