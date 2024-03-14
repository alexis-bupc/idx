import json

json_data = '''
{
  "description": "This is a JSON string with a\\nnew line character."
}
'''

data = json.loads(json_data)
print(data["description"])