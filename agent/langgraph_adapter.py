class LangGraphUnavailable(RuntimeError): pass
class LangGraphAgentRunner:
    def __init__(self, agent_graph): self.agent_graph = agent_graph
    def invoke(self, state, config=None):
        p = {k: v for k, v in dict(state or {}).items() if not str(k).startswith('_')}
        s = self.agent_graph.run(**p)
        p['_agent_state'] = s
        return p
