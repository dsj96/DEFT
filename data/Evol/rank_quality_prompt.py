base_instruction = """Rank the following responses provided by different AI assistants to the user's question according to the quality of their responses.
Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
The following are {num} responses, each indicated by number identifier [].
You can rank them based on their relevance to the question: {question}

{responses}

The user's question is: 

{question}

You will rank the {num} responses above based on their relevance to the question.
The responses will be listed in descending order using identifiers, and the most relevant response should be listed first, and the output format should be [] > [] > etc, e.g., [1] > [2] > etc.
The ranking results of the {num} responses (only identifiers) is:
"""




def createRankQualityPrompt(question, num, responses, chat_mode=False):
    if not chat_mode:
        
        prompt = base_instruction.format(question=question, num=num, responses='\n'.join(responses))
        
        history=[{"role": "system", "content": "You are RankGPT, an intelligent assistant that can rank responses based on their questions."}, {"role": "user", "content": prompt}]
    
    else:
        history=[
            {"role": "system", "content": "You are RankGPT, an intelligent assistant that can rank responses based on their questions."},
            {"role": "user", "content": "I will provide you with {num} responses, each indicated by number identifier []. Rank them based on their relevance to question: {question}.\nYour evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.".format(num, question)},
            {"role": "assistant", "content": "Okay, please provide the responses."}
        ]

        user_assistant_history = []
        for index,item in enumerate(responses):
            user_assistant_history.append(
               {"role": "user", "content": item}
            )
            user_assistant_history.append(
               {"role": "assistant", "content": "Received responses [{}]".format(index+1)}
            )
        history = history + user_assistant_history + [{
            "role": "assistant", "content": "Search question: {question}.\nRank the {num} responses above based on their relevance to the search question. The responses should be listed in descending order using identifiers, and the most relevant response should be listed first, and the output format should be [] > [], e.g., [1] > [2]. Only return the ranking results, do not say any word or explain.".format(question=question, num=num)
        }]
        
    return history