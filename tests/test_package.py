import agent
import agent.tools


def test_packages_import():
    assert agent.__doc__
    assert agent.tools.__doc__
