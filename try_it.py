from translation_fidelity import score

result = score("Hello, how are you?", "Ciao, come stai?", "it")
print(result.to_dict())

result = score("I love pizza.", "Odio la pizza.", "it")
print(result.to_dict())

result = score("Hello, how are you?", "Bonjour, comment allez-vous?", "it")
print(result.to_dict())