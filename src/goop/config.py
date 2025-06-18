import os
import json
import re


with open(os.path.join(os.path.dirname(__file__), "mcp_config.json"), "r") as f:
    mcp_config = json.load(f)


def resolve_env_vars(confg: dict):
    for server, server_config in confg["mcpServers"].items():
        if "env" in server_config:
            for env_var in server_config["env"]:
                confg["mcpServers"][server]["env"][env_var] = os.environ.get(env_var, "")
                if confg["mcpServers"][server]["env"][env_var] == "":
                    raise ValueError(f"Environment variable {env_var} is not set")
        if "args" in server_config:
            for i, arg in enumerate(server_config["args"]):
                # Handle ${VAR} patterns anywhere in the string
                def replace_env_var(match):
                    env_var = match.group(1)
                    env_value = os.environ.get(env_var, "")
                    if env_value == "":
                        raise ValueError(f"Environment variable {env_var} is not set")
                    return env_value

                # Replace all ${VAR} patterns in the string
                confg["mcpServers"][server]["args"][i] = re.sub(r'\$\{([^}]+)\}', replace_env_var, arg)
    return confg


mcp_config = resolve_env_vars(mcp_config)
