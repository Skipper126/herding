

class AgentsLayout:

    def __init__(self, env_data):
        pass

    @staticmethod
    def get_function(agent_layout):
        return{
            constants.AgentLayout.RANDOM : AgentLayoutFunction._random,
            constants.AgentLayout.LAYOUT1 : AgentLayoutFunction._layout1,
            constants.AgentLayout.LAYOUT2 : AgentLayoutFunction._layout2
        }[agent_layout]

    @staticmethod
    def _random(env):
        padding = 5
        for agent in env.dog_list + env.sheep_list:
            x = random.randint(agent.radius + padding, env.map_width - agent.radius - padding)
            y = random.randint(agent.radius + padding, env.map_height - agent.radius - padding)
            agent.set_pos(x, y)

    @staticmethod
    def _layout1(env):
        sheep_padding = 5
        for agent in env.sheep_list:
            x = random.randint(agent.radius + sheep_padding, env.map_width - agent.radius - sheep_padding)
            y = random.randint(agent.radius + sheep_padding + 200, env.map_height - agent.radius - sheep_padding)
            agent.set_pos(x, y)

        for i, agent in enumerate(env.dog_list):
            x = (i + 1) * (env.map_width / (env.dog_count + 1))
            y = 0
            agent.set_pos(x, y)

    @staticmethod
    def _layout2(env):
        # TODO
        pass
