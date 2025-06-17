import os
import json


with open(os.path.join(os.path.dirname(__file__), "mcp_config.json"), "r") as f:
    mcp_config = json.load(f)


def resolve_env_vars(confg: dict):
    for server, server_config in confg["mcpServers"].items():
        if "env" in server_config:
            for env_var in server_config["env"]:
                confg["mcpServers"][server]["env"][env_var] = os.environ.get(env_var, "")
                if confg["mcpServers"][server]["env"][env_var] == "":
                    raise ValueError(f"Environment variable {env_var} is not set")
    return mcp_config


mcp_config = resolve_env_vars(mcp_config)
