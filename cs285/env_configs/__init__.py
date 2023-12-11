from .dqn_atari_config import atari_dqn_config
from .dqn_basic_config import basic_dqn_config
from .sac_config import sac_config
from .sac_config_new import sac_config_new
from .ppo_config import ppo_config

configs = {
    "dqn_atari": atari_dqn_config,
    "dqn_basic": basic_dqn_config,
    "sac": sac_config,
    "sac_new": sac_config_new,
    "ppo": ppo_config,
}
