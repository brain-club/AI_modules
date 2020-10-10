# Using library from https://github.com/vzhou842/profanity-check

from profanity_check import predict_prob

def censoring(input_list):

    '''
		INPUT:
            input_list : (list of sentences) list of sentences want to check
        OUTPUT:
            predictions : (list of booleans) list consists of true/false if ther sentence includes offensiveness or profanity
    '''

    threshold = 0.15
    predictions = []

    predictions = [prob>threshold for prob in predict_prob(input_list)]

    return predictions


if __name__ == "__main__":
    
    print("Starting demo")

    input_sent = input("your input sentence : ")
    print(f"Result : {censoring([input_sent])} with a profanity probability of {predict_prob([input_sent])}")
