import re
import math
from typing import Dict
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


###accuracy reward
"""
1. Transform the solution into a structured mathematical representation using latex2sympy2.
2. If parsing is unsuccessful, allocate a neutral score of 0.5.
3. Retrieve the model output and standardize it for improved robustness.
4. Utilize math_verify to determine whether the parsed response aligns with the parsed solution.
5. Assign a score of 1 if correct, otherwise assign 0.
"""

def accuracy_reward(completions,**kwargs):
    contents=[completion[0]["content"] for completion in completions]
    rewards=[]
    solutions=kwargs.get("solution")
    if solutions is None:
        return [0.5] * len(completions) #netural reward if no solution
    for content, sol in zip(contents,solutions):
        #parse the ground truth solution
        gold_parsed=parse(sol,
                          extraction_node="first_match",
                          extraction_config=LatexExtractionConfig())
        if len(gold_parsed) != 0:
            answer_parsed=parse( #parse the models answer
                content,
                extraction_mode="first_match",
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ]
            )
            try:
                reward=float(verify(answer_parsed,gold_parsed)) #if its correct reward 1 if not 0
            except:
                reward=0.0       
                print(f"verify failed truth: {gold_parsed} answer: {answer_parsed}")
        else:
            reward=1 # if the ground truth cant be parsed(neutral reward 1)
            print(f"failed to parse gold solution: ", sol)
        rewards.append(reward)
    return rewards


###format reward
"""
1. We define a regular expression (regex) pattern that specifies the expected structure: the content should begin with a certain part, contain anything until a specific point, include spaces, continue with another section, have more content until another marker, and then end precisely there.
2. We extract the actual text content from each model’s completion.
3. We apply re.match to check if the extracted content strictly follows the defined pattern. The re.DOTALL flag ensures that . matches newlines, while re.MULTILINE allows ^ and $ to represent the start and end of the entire string rather than individual lines.
4. Finally, we assign a reward of 1 if the format is matched perfectly and 0 if it is not, enforcing a strict binary reward for format correctness."""

def format_reward(completions,**kwargs):
    pattern="^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents=[completion[0]["content"] for completion in completions]
    matches=[re.match(pattern, content ,re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


###reasoning step reward
"""
We will identify keywords and patterns such as:
1. Step-based sequences – Phrases like Step 1, Step 2, etc.
2. Numbered lists – Sequences using numbers like 1, 2, 3,...
3. Bullet points – Indicators such as - or **.
4. Transition words – Terms like First, Second, Next, Finally that signal logical progression.
"""

def reasoning_steps_reward(completions,**kwargs):
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_content=[completion[0]["content"] for completion in completions]
    matches=[re.match(pattern,content,re.MULTILINE) for content in completion_content]
    return [min(1.0 ,count / 3 ) for count in matches] #encouraging atleast 3 steps


###Cosine Scaled reward
""" 
For correct answers, we prioritize concise and direct solutions over lengthy, overly detailed ones. A brief, accurate response is often more effective.
For incorrect answers, a short, incorrect response is generally worse than a longer one that at least attempts reasoning. Therefore, we apply a stricter penalty to short incorrect answers while being more lenient with longer ones that show effort.
"""

def get_cosine_scaled_reward(min_value_wrong: float=-0.5,
                             max_value_wrong: float=-0.1,
                             min_value_correct: float=0.8,
                             max_value_correct: float=1.0,
                             max_len: int=1000):
    def cosine_scaled_reward(completions,solution,accuracy_reward, **kwargs):
        contents=[completions[0]["content"] for completion in completions]
        rewards=[]
        for content,sol,acc_reward in zip(contents,solution,accuracy_reward):
            gen_len=len(content)
            progress=gen_len/max_len #how far we are from generated answer
            cosine=math.cos(progress*math.pi) # cos value
            if acc_reward >0.5:
                min_value=min_value_correct
                max_value=max_value_correct
            else:
                min_value=min_value_wrong
                max_value=max_value_wrong
            reward=min_value + 0.5 * (max_value - min_value) *(1.0+cosine) #cosine scaling formula
            rewards.append(float(reward))
        return rewards
    return cosine_scaled_reward


###repetition penality reward
""" 
discouraging the model from getting stuck in the loop
if it uses the same seq of words(ngram) too many times
"""

def get_repetition_penaltiy_reward(ngram_size: int=3, max_penalty: float =-0.1):
    if max_penalty >0:
        raise ValueError(f"max_penality {max_penalty} should not be postivie")
    def zipngram(text: str,ngram_size: int):
        words=text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])
    def repetition_penalty_reward(completions,**kwarg) -> float:
        contents=[completion[0]["content"] for completion in completions]
        rewards=[]
        for completion in contents:
            if completion=="": # no penality for empty completion
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size: #no penality for short completion
                rewards.append(0.0)
                continue
            ngram=set() #store uniqe unigram
            total=0
            for ng in zipngram(completion,ngram_size):
                ngram.add(ng)
                total+=1
            scaling=1-len(ngram)/total #more repetition higher scaling
            reward=scaling * max_penalty
            rewards.append(reward)
        return rewards
    return repetition_penalty_reward


###xml tag counting reward
"""
this checks if we generate the exact number of <think> and <answer> xml tag
its quite similar with the foramat reward.
"""

def xml_count_reward(completions,**kwarg):
    def count_xml(text: str) -> float:
        count=0.0
        if text.count("<think>\n") ==1:
            count +=0.25
        if text.count("\n</think>\n") ==1:
            count += 0.25
        if text.count("\n<answer>\n")==1:
            count += 0.25
        if text.count("\n</answer>") ==1:
            count += 0.25
        return count
    contents=[completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


#length of reward
"""
dicourages overthinking during generation
generate token efficently 
"""

def len_reward(completions: list[Dict[str,str]],solution: list[str],**kwargs) -> float:
    contents=[completion[0]["content"] for completion in completions]
    correctness=[]
    for content, sol in zip(contents,solution):
        gold_parse=parse(
            sol,
            extraction_config=[LatexExtractionConfig()],
            extraction_mode="first_match"
        )
        if len(gold_parse)==0:
            correctness.append(True) #skip unparsable examples by treating them as a true value
            print("failed to parse gold solution",sol)
            continue
        answer_parse=parse(
            content,
            extraction_mode="first_match",
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True
                    )
                )
            ]
        )
        correctness.append(verify(answer_parse,gold_parse))
    lengths=[len(content) for content in contents]
    min_len=min(lengths)
    max_len=max(lengths)

    if max_len == min_len:
        return [0.0 * len(completions)] #just return zero
    rewards=[]
    for length,is_correct in zip(length,correctness):
        lambda_val=0.5-(length-min_len)/(max_len-min_len) #scaling
        if is_correct:
            reward=lambda_val
        else:
            reward=min(0,lambda_val)
        rewards.append(float(reward))
    return rewards

#TODO implement reward for generated code.