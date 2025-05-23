from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()
for tool in tool_spec_list:
    print(tool.metadata.name, tool.metadata.description) 
