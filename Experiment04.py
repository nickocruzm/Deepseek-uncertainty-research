import requests
import time
import csv
from typing import List
from collections import Counter, defaultdict
import math
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

"""
MULT-LABEL queries with no distractions injected.
"""

# DeepSeek API Configuration
API_KEY = "NEED API-KEY TO RUN"
API_URL = 'https://api.deepseek.com/v1/chat/completions'
MODEL = 'deepseek-chat'  # Adjust based on your subscription

HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Parameters
REPEAT_COUNT = 10
OUTPUT_FILE = 'deepseek_v3_mi_estimate.csv'

# ---- Prompt Constructor Injects uncertainty ----
def construct_prompt(query: str, prev_answers: List[str], distractors: dict = None, iteration: int = 0) -> str:
    prompt = f"Consider the following question:\nQ: {query}\n"

    # Inject distractors only at iteration 1 (optional)
    if distractors and iteration == 1 and query in distractors:
        for distractor in distractors[query]:
            prompt += f"Another answer to question Q is: {distractor}\n"

    if prev_answers:
        prompt += f"Another answer to question Q is: {prev_answers[-1]}\n"

    prompt += f"Provide an answer to the following question:\nQ: {query}\nA:"
    return prompt

# ---- Query DeepSeek-v3 ----
def query_deepseek(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return "ERROR"
    
    return response.json()['choices'][0]['message']['content'].strip()

    


def mutual_information_estimate_fuzzy(responses, similarity_threshold=85):
    # --- Step 1: Cluster similar responses ---
    clusters = []
    for response in responses:
        matched = False
        for cluster in clusters:
            if fuzz.ratio(response, cluster[0]) >= similarity_threshold:
                cluster.append(response)
                matched = True
                break
        if not matched:
            clusters.append([response])

    # --- Step 2: Get cluster sizes (empirical probabilities) ---
    total = len(responses)
    cluster_sizes = [len(c) for c in clusters]
    probs = [size / total for size in cluster_sizes]

    # --- Step 3: Compute entropy-based MI lower bound ---
    mi = sum(p * math.log(p / (1 / len(probs))) for p in probs if p > 0)
    return mi
# ---- Run Main Experiment ----


def run_deepseek_experiment(queries: List[str]):
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query", "Y1", "Y2", "MutualInformation", "AllResponses"])

        for query in queries:
            print(f"Processing: {query}")
            responses = []
            for i in range(REPEAT_COUNT):
                prompt = construct_prompt(query, responses, distractors=None, iteration=i)
                print(f"\tPROMPT {i}: {prompt} \n")
                response = query_deepseek(prompt)
                responses.append(response)
                print(f'RESPONSE: {response}\n')
                time.sleep(1.2)  # Respect rate limit
                

            #mi = mutual_information_estimate(responses)
            mi = mutual_information_estimate_fuzzy(responses, similarity_threshold=85)
            writer.writerow([query, responses[0], responses[1] if len(responses) > 1 else "", f"{mi:.4f}", "|".join(responses)])
            print(f"→ MI = {mi:.4f} for query: {query}")


queries = [
    "Name a city in the UK.",
    "Name a yellow fruit.",
    "Name an alcoholic drink.",
    "Name a ball game that is played by more than 5 players."
]

distractors = {
    "Name a city in the UK.": [
        "A city in the UK is Edinburgh."
    ],
    "Name a yellow fruit.": [
        "A yellow fruit is Mango."
    ],
    "Name an alcoholic drink.": [
        "An alcoholic drink is Tequila."
    ],
    "Name a ball game that is played by more than 5 players.": [
        "A ball game played by more than 5 players is Rugby."
    ]
}

run_deepseek_experiment(queries)