import random
import requests
import json
import os

url = 'https://opentdb.com/api.php?amount=50&type=multiple'


def check_internet_connection(url='http://www.google.com'):
    try:
        res = requests.get(url, timeout=5)
        return True
    except requests.ConnectionError:
        return False


def read_questions():
    with open('questions.json', 'r') as file:
        list = json.load(file)
        index = 0
        quizz = []
        random_numbers = random.sample(range(50), 10)

        for question in list:
            if index in random_numbers:
                quizz.append(question)
                index += 1
            else:
                index += 1

    return quizz


def fetching():
    if check_internet_connection():
        response = requests.get(url)

        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            questions = data['results']

            with open('questions.json', 'w') as json_file:
                json.dump(questions, json_file, indent=4)
            return read_questions()


        else:
            print(f"Failed to retrieve questions. Status code: {response.status_code}")
    else:
        if os.stat('questions.json').st_size == 0:
            print('No questions available')
        else:
           return read_questions()







