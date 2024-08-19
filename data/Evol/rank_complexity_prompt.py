base_instruction = """Ranking the following instructions provided by different users according to the instruction-following difficulty and complexity.
The following are {num} instructions, each indicated by number identifier [].
The user's instructions are: 

{instructions}

You will rank the {num} instructions above based on the instruction-following difficulty and complexity.
The instructions will be listed in descending order using identifiers, and the most relevant response should be listed first, and the output format should be [] > [] > etc, e.g., [1] > [2] > etc.
The ranking results of the {num} responses (only identifiers) is:
"""




def createRankComplexityPrompt(instructions, num, chat_mode=False):
    if not chat_mode:
        
        prompt = base_instruction.format(instructions='\n'.join(instructions), num=num)
        
        history=[{"role": "system", "content": "You are RankGPT, an intelligent assistant that can rank instructions based on their instruction-following difficulty and complexity."}, {"role": "user", "content": prompt}]
    
    else:
        history=[
            {"role": "system", "content": "You are RankGPT, an intelligent assistant that can rank instructions based on the instruction-following difficulty and complexity."},
            {"role": "user", "content": "I will provide you with {num} instructions, each indicated by number identifier []. Rank them based on their instruction-following difficulty and complexity.".format(num)},
            {"role": "assistant", "content": "Okay, please provide the instructions."}
        ]

        user_assistant_history = []
        for index,item in enumerate(instructions):
            user_assistant_history.append(
               {"role": "user", "content": item}
            )
            user_assistant_history.append(
               {"role": "assistant", "content": "Received instructions [{}]".format(index+1)}
            )
        history = history + user_assistant_history + [{
            "role": "assistant", "content": "Rank the {num} instructions above based on their instruction-following difficulty and complexity. The instructions should be listed in descending order using identifiers, and the most complex and difficult instruction should be listed first, and the output format should be [] > [], e.g., [1] > [2]. Only return the ranking results, do not say any word or explain.".format(num=num)
        }]
        
    return history