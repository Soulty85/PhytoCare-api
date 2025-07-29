from dotenv import load_dotenv

import argostranslate.package
import argostranslate.translate
import json
import os 
import requests


load_dotenv()

from_code = "en"
to_code = "fr"
API_KEY = os.getenv("OPEN_ROUTER_API_KEY").strip('"')

# Download and install Argos Translate package
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())

# formate la predicton du model
def format_output(prediction: str) -> str:
    dessease = prediction.split(" ")
    delimiter = " "
    
    if len(dessease) <= 2:
        return dessease[-1]
    
    return delimiter.join(dessease[-2:])

# traduit le nom de la maladie en francais
def translate_dease(desease: str) -> str:
    translatedText = argostranslate.translate.translate(desease, from_code, to_code)
    
    return translatedText

# donne des methodes de traitement pour traiter la maladie
def get_advice(desease: str) -> str:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "google/gemma-3n-e2b-it:free",
            "messages": [
                {
                    "role": "user",
                    "content": f"En un paragraphe comment traiter cette maladie des plantes: {desease}"
                }
            ],
            
        })
    )
    
    return response.json()

healthy_advice = "Pour bien traiter une plante saine, il faut lui offrir de bonnes conditions de vie afin de prévenir les maladies. Cela passe par un arrosage régulier mais modéré, une exposition adaptée à la lumière, un sol bien drainé, et un apport d’engrais naturel pendant la période de croissance. Il est aussi important de retirer les feuilles mortes, de nettoyer les outils utilisés et d’observer régulièrement la plante pour détecter rapidement tout signe de problème. En complément, on peut utiliser des traitements préventifs naturels comme du purin d’ortie ou du savon noir pour renforcer la plante et éloigner les parasites."