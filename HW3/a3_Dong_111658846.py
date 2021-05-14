import numpy as np
from sys import argv
import sys
import re

sys.stdout = open('a3_Dong_111658846_OUTPUT.txt', 'w',encoding='utf-8')

def loadData(path,aDict):
    """
    this function will read data and generate a dictionary of words and their counts
    :param path: A path of training file
    :param aDict: take a dictionary and fill it up
    :return: return a list of sentances
    """
    f = open(path,'r',encoding='utf-8')
    result = []
    for i in f.readlines():
        # get only sentence part
        sentence = i.split()[2:]
        headMatch = re.compile(r'<head>([^<]+)</head>')

        # delete head
        for x in range(len(sentence)):
            # get rid of \n
            if "\n" in sentence[x]:
                sentence[x] = sentence[x].replace("\n", "")

            m = headMatch.match(sentence[x])
            if m:  # a match: we are at the target token
                sentence[x] = m.groups()[0]

        # delete lemma and POS
        for j in range(len(sentence)):
            sentence[j] = str(sentence[j]).split("/")[0]

        # insert <s> </s> to sentence
        sentence.insert(0,"<s>")
        sentence.append("</s>")

        #lower case all word and create dictionary
        for j in range(len(sentence)):
            # lower case
            if sentence[j].isalpha():
                sentence[j] = sentence[j].lower()

            # add to dict
            if sentence[j] in dictionary:
                dictionary[sentence[j]] += 1
            else:
                dictionary[sentence[j]] = 1

        # concate to sentence and add to overall array
        result.append(sentence)

    return result

def uniBiTri(dictionary, trainDict, sentences):
    """
    this function will examine through sentences and generate 1/2/3 grams count
    :param dictionary: take all word counts dictionary
    :param trainDict: a list of 5000 word
    :param sentences: sentences to create 1/2/3 grams from
    :return:
    """
    uni = {}
    bi = {}
    tri = {}

    # get unigram
    for k,v in dictionary.items():
        if k in trainDict:
            uni[k] = v
        else:
            if "<OOV>" in uni:
                uni["<OOV>"] += v
            else:
                uni["<OOV>"] = v

    # get bigram
    for j in sentences:
        for i in range(len(j)-1):
            first = j[i]
            second = j[i+1]
            if first not in trainDict:
                first = "<OOV>"
            if second not in trainDict:
                second = "<OOV>"

            if first in bi:
                if second in bi[first]:
                    bi[first][second] += 1
                else:
                    bi[first][second] = 1
            else:
                bi[first] = {}
                bi[first][second] = 1

    # get trigram
    for j in sentences:
        for i in range(len(j)-2):
            first = (-1,-1)
            if j[i] in trainDict and j[i+1] in trainDict:
                first = (j[i],j[i+1])
            elif j[i] in trainDict and j[i+1] not in trainDict:
                first = (j[i], "<OOV>")
            elif j[i] not in trainDict and j[i+1] in trainDict:
                first = ("<OOV>",j[i+1])
            elif j[i] not in trainDict and j[i+1] not in trainDict:
                first = ("<OOV>","<OOV>")

            second = ""
            if j[i+2] in trainDict:
                second = j[i+2]
            else:
                second = "<OOV>"

            if first in tri:
                if second in tri[first]:
                    tri[first][second] += 1
                else:
                    tri[first][second] = 1
            else:
                tri[first] = {}
                tri[first][second] = 1

    return uni,bi,tri


def prob(wordMinus1, wordMinus2 = None):
    """
    this function will get probability of both bigram and trigram
    :param wordMinus1: wi-1
    :param wordMinus2: wi-2
    :return: a dictionary of probabilities
    """

    # bigram
    if wordMinus2 == None:
        aDict = {}
        total = np.array(list(bigramCounts[wordMinus1].values())).sum()+5000 # add 1 smoothing
        for k,v in bigramCounts[wordMinus1].items():
            aDict[k] = (bigramCounts[wordMinus1][k]+1)/total
        return aDict

    else: # trigram
        wiLst = list(bigramCounts[wordMinus1].keys())
        aDict = {}
        total = np.array(list(bigramCounts[wordMinus1].values())).sum()+5000
        firstProb = {}
        for k, v in bigramCounts[wordMinus1].items():
            firstProb[k] = (bigramCounts[wordMinus1][k] + 1) / total
        if "<OOV>" not in firstProb:
            firstProb["<OOV>"] = (bigramCounts[wordMinus2].get("<OOV>",0) + 1) / total
        # calculate interpolation
        try:
            total = np.array(list(trigramCounts[(wordMinus2,wordMinus1)].values())).sum() + 5000
            for k,v in trigramCounts[(wordMinus2,wordMinus1)].items():
                aDict[k] = (firstProb[k] + \
                           (trigramCounts[(wordMinus2,wordMinus1)][k]+1)/total)/2
        except KeyError:
            return firstProb
        # add word in wilst but not in aDict into aDict
        for i in wiLst:
            if i not in aDict:
                aDict[i] = firstProb["<OOV>"]/2

        return aDict

def grnerate(lst = []):
    """
    this function generate max length of 32 word sentences
    :param lst: initial word list which use to generate, default is empty list
    :return: a list of generated sentences
    """
    if len(lst) == 0: # if list is empty
        return None
    if len(lst) == 1: # if list have one word
        next = prob(lst[0])
        word = np.random.choice([i for i in list(next.keys())], p=[i/np.array(list(next.values())).sum() for i in list(next.values())])
        lst.append(word)

    # do trigrams on list and generate next word
    for i in range(32-len(lst)):
        if lst[-1] == "</s>":
            break
        next = prob(lst[-1],lst[-2])
        word = np.random.choice([j for j in list(next.keys())], p=[j/np.array(list(next.values())).sum() for j in list(next.values())])
        lst.append(word)

    return lst


# python a3_Dong_111658846.py onesec_train.tsv

if __name__ == '__main__':
    if len(argv)!=2:
        print("Usage: python3 a3_LASTNAME_ID.py onesec_train.tsv")
        exit(-1)

    path = argv[1]

    dictionary = {}
    x = loadData(path,dictionary)

    # get 5000 top frequent words
    trainDict = list(dict(sorted(dictionary.items(), key=lambda j: (j[1], j[0]), reverse=True)[:5000]).keys())

    # get unigram, bigram, trigram counts
    global unigramCounts
    global bigramCounts
    global trigramCounts
    unigramCounts,bigramCounts,trigramCounts = \
        uniBiTri(dictionary,trainDict,x)

    # Part 2.2 Check
    print("CHECKPOINT 2.2 - counts")
    print("1 grams")
    print("('language',)",unigramCounts['language'])
    print("('the',)",unigramCounts['the'])
    print("('formal',)",unigramCounts['formal'])

    print("2 grams")
    print("('the', 'language')",bigramCounts['the']['language'])
    print("('<OOV>', 'language')",bigramCounts['<OOV>']['language'])
    print("('to', 'process')",bigramCounts['to']['process'])

    print("3 grams")
    print("('specific', 'formal', 'languages')",trigramCounts[('specific','formal')]['languages'])
    print("('to', 'process', '<OOV>')",trigramCounts.get(('to','process'),0).get('<OOV>',0)) # Can't find key cause there is no such thing
    print("('specific', 'formal', 'event')",trigramCounts.get(('specific','formal'),0).get('event',0)) # Can't find key cause there is no such thing

    # Part 2.3 Check
    print("\nCHECKPOINT 2.3 - Probs with addone")
    print("2 grams:")
    print("('the', 'language')",prob('the')['language'])
    print("('<OOV>', 'language')",prob('<OOV>')['language'])
    print("('to', 'process')",prob('to')['process'])

    print("3 grams:")

    print("('specific', 'formal', 'languages')",prob('formal','specific').get('languages',"Not valid Wi"))
    print("('to', 'process', '<OOV>')",prob('process','to').get('<OOV>',"Not valid Wi"))
    print("('specific', 'formal', 'event')",prob('formal','specific').get('event',"Not valid Wi"))

    # Part 2.4 Check
    print("\nFINAL CHECKPOINT - Generated Language")

    print("\nPROMPT: <s>")
    print("    "," ".join(grnerate(["<s>"])))
    print("    "," ".join(grnerate(["<s>"])))
    print("    "," ".join(grnerate(["<s>"])))

    print("\nPROMPT: <s> language is")
    print("    "," ".join(grnerate(["<s>","language","is"])))
    print("    "," ".join(grnerate(["<s>", "language", "is"])))
    print("    "," ".join(grnerate(["<s>", "language", "is"])))

    print("\nPROMPT: <s> machines")
    print("    "," ".join(grnerate(["<s>", "machines"])))
    print("    "," ".join(grnerate(["<s>", "machines"])))
    print("    "," ".join(grnerate(["<s>", "machines"])))

    print("\nPROMPT: <s> they want to process")
    print("    "," ".join(grnerate(["<s>", "they","want","to","process"])))
    print("    "," ".join(grnerate(["<s>", "they", "want", "to", "process"])))
    print("    "," ".join(grnerate(["<s>", "they", "want", "to", "process"])))

