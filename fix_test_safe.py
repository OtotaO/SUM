import os

file_path = 'Tests/test_api_endpoints.py'
with open(file_path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "app.config['TESTING'] = True" in line:
        indent = line[:line.find("app.config")]
        new_lines.append(f"{indent}app = create_simple_app()\n")
        new_lines.append(line)
    elif "key_data = create_api_key(" in line:
        indent = line[:line.find("key_data")]
        # Extract arguments
        args = line.split("create_api_key(")[1].strip().rstrip(')')
        new_lines.append(f"{indent}manager = get_auth_manager()\n")
        new_lines.append(f"{indent}_, key_secret = manager.generate_api_key({args})\n")
        new_lines.append(f"{indent}key_data = {{'api_key': key_secret}}\n") # Mock the dict structure expected by next lines
    else:
        new_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(new_lines)
