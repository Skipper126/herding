from .agents_controller import AgentsController
from ..data import EnvData
from ..configuration.names import ConfigName as cn

def create_agents_controller(env_data: EnvData) -> AgentsController:
    if env_data.config[cn.HARDWARE_ACCELERATION] is True:
        from .gpu.gpu_agents_controller import GpuAgentsController
        return GpuAgentsController(env_data)
    else:
        from .cpu.cpu_agents_controller import CpuAgentsController
        return CpuAgentsController(env_data)
