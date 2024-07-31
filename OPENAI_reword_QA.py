""" 
John Beggs // JULY2024 // TRAC AIAD
Run this script using combined_qa.parquet as the input file.

You'll also need an OpenAI API key - approximate cost is 10 cents and should take less than an hour to run on good wifi

Outputted file is reword_QA.parquet - this has taken the combined Q/A column, split the two, plugged
both into the LLM for re-wording, then created separate columns

Potential problem: final size of df is 9590 x 6, not 9600
"""

import openai
import argparse
import pandas as pd
from tqdm import tqdm

# openai.api_key = !!! api key here !!!

def reword_q_a(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Reword this question and answer pair: {text}"}
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()

def rewordQnA(df):
    reworded_questions = []
    reworded_answers = []

    for i in tqdm(range(len(df))):
        q_a_pair = df["Q_A"].iloc[i]
        reworded_q_a = reword_q_a(q_a_pair)

        if ':' in reworded_q_a:
            reworded_question, reworded_answer = reworded_q_a.split(':', 1)
            reworded_questions.append(reworded_question.strip())
            reworded_answers.append(reworded_answer.strip())
        else:
            question, answer = q_a_pair.split(':', 1)
            reworded_questions.append(question.strip())
            reworded_answers.append(answer.strip())

        if i == 0:
            print(f"First reworded Q_A: {reworded_q_a}")

    return reworded_questions, reworded_answers

def augmentQnA(path):
    df = pd.read_parquet(path)
    # df = df.iloc[0:4800]
    print(f"Original DataFrame shape: {df.shape}")

    reworded_questions, reworded_answers = rewordQnA(df)
    print("Finished re-wording questions and answers using LLM.")

    df['Question'] = reworded_questions
    df['Answer'] = reworded_answers
    df.drop(columns=['Q_A'], inplace=True)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script re-words questions and answers in the dataset.")
    parser.add_argument("--path", required=True, type=str)

    args = parser.parse_args()

    training_set = augmentQnA(args.path)
    training_set.to_parquet('reword_QA.parquet', compression='BROTLI')

    print("Done.")