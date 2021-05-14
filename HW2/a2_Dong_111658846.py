import sys
import torch
import torch.nn as nn
import numpy as np
import re
import math


sys.stdout = open('a2_Dong_111658846_OUTPUT.txt', 'w') # remove common once complete whole work

# Part 1.1
def loadData(filename,dicttionary):
    f = open(filename)
    process = []
    machine = []
    language = []

    # classify to each list
    f = open(filename,encoding='utf-8')
    for line in f.readlines():
        if "process." in line:
            process.append(line)
        if "machine." in line:
            machine.append(line)
        if "language." in line:
            language.append(line)

    # replace string by int
    for name in (process,machine,language):
        for i in range(len(name)):
            name[i] = name[i].split("\t")
            name[i][0] = int(name[i][0].split(".")[2])
            #name[i][1] = name[i][1].split("%")[1].split(":")
            #name[i][1] = int(name[i][1][0])+int(name[i][1][1])+int(name[i][1][2])
            name[i][2] = name[i][2].split(" ")

            # get rid of <head>
            headMatch = re.compile(r'<head>([^<]+)</head>')
            headIndex = -1
            for x in range(len(name[i][2])):
                # get rid of \n
                if "\n" in name[i][2][x]:
                    name[i][2][x] = name[i][2][x].replace("\n","")

                m = headMatch.match(name[i][2][x])
                if m:  # a match: we are at the target token
                    name[i][2][x] = m.groups()[0]
                    headIndex = x

            # split by /
            for j in range(len(name[i][2])):
                name[i][2][j] = name[i][2][j].split("/")
                # lower case pure character strings
                if name[i][2][j][0].isalpha():
                    name[i][2][j][0] = name[i][2][j][0].lower()
                #record vocab in dict
                if name[i][2][j][0] in dicttionary:
                    dicttionary[name[i][2][j][0]] += 1
                else:
                    dicttionary[name[i][2][j][0]] = 1

    return process,machine,language

# This function return a matrix with wordCount * 4000
# Part 1.2
def embedding(list, vocab, label):
    prev = [0 for j in range(len(vocab))]
    after = [0 for j in range(len(vocab))]
    for i in range(len(list)):
        if isinstance(list[i][0],str):
            if list[i][0].lower() == label:
                try:
                    index1 = vocab.index(list[i-1][0])
                    prev[index1] = 1
                    index1 = vocab.index(list[i+1][0])
                    after[index1] = 1
                except ValueError:
                    pass

    prev.extend(after)
    return prev

def labelEncode(samples):
    labels = list(set([i[1] for i in samples]))
    result = []
    for i in samples:
        result.append(labels.index(i[1]))
    return result

# Part 1.3
class LogReg(nn.Module):
    def __init__(self, num_feats,classes, learn_rate = 0.01, device = torch.device("cpu") ):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(num_feats+1, classes)

    def forward(self, X):
        newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1)
        #return 1/(1 + torch.exp(-self.linear(newX)))
        return self.linear(newX)

# Part 2.1
def cooccurrence(process,machine,language,dictTrain):
    sentences = []
    for i in process:
        sentence = []
        for j in i[2]:
            sentence.append(j[0])
        sentences.append(sentence)
    for i in machine:
        sentence = []
        for j in i[2]:
            sentence.append(j[0])
        sentences.append(sentence)
    for i in language:
        sentence = []
        for j in i[2]:
            sentence.append(j[0])
        sentences.append(sentence)

    coMatrix = [[0 for j in range(len(dictTrain)+1)] for i in range(len(dictTrain)+1)]

    for i in sentences:
        for j in i:
            for k in i:
                word1 = -1
                word2 = -1
                try:
                    word1 = dictTrain.index(j)
                except:
                    pass
                try:
                    word2 = dictTrain.index(k)
                except:
                    pass
                coMatrix[word1][word2] += 1
    return coMatrix

# Part 2.3
def euDistance(vec1,vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sqrt(np.sum(np.square(vec1-vec2)))

# Part 3.1
def PCAembedding(lst,Udict,target):
    result = []
    for i in range(len(lst)):
        if lst[i][0].lower() == target:
            prev1 = prev2 = after1 = after2 = []
            try:
                prev1 = Udict[lst[i-2][0]]
            except:
                prev1 = Udict["OOV"]
            try:
                prev2 = Udict[lst[i-1][0]]
            except:
                prev2 = Udict["OOV"]
            try:
                after1 = Udict[lst[i+1][0]]
            except:
                after1 = Udict["OOV"]
            try:
                after2 = Udict[lst[i+2][0]]
            except:
                after2 = Udict["OOV"]
            result.extend(prev1)
            result.extend(prev2)
            result.extend(after1)
            result.extend(after2)
            break

    if not result:
        return [0 for i in range(200)]
    return result


# python a2_Dong_111658846.py onesec_train.tsv onesec_test.tsv

# python a2_Dong_111658846.py onesec_test.tsv onesec_test.tsv

if __name__ == "__main__":
    print("\nLOADING DATA...")
    if len(sys.argv) != 3:
        print("USAGE: python3 a2_lastname_id.py onesec_train.tsv onesec_test.tsv")
        sys.exit(1)

    filename = sys.argv[1]
    testName = sys.argv[2]

    # Part 1.1 read data and generate dictionary
    dictTrain,dictTest = {},{}
    process, machine, language = loadData(filename,dictTrain)
    processTest, machineTest, languageTest = loadData(testName, dictTest)

    dictTrain = list(dict(sorted(dictTrain.items(), key = lambda x:(x[1],x[0]), reverse=True)[:2000]).keys())

    # Part 1.2 do double one-hot embedding
    processClasses = len(set([i[1] for i in process]))
    machineClasses = len(set([i[1] for i in machine]))
    languageClasses = len(set([i[1] for i in language]))
    print("Extract X and Y from train set: ")
    processX = torch.from_numpy(np.array([embedding(i[2], dictTrain,"process") for i in process]).astype(np.float32))
    processY = torch.from_numpy(np.array(labelEncode(process))).long()
    machineX = torch.from_numpy(np.array([embedding(i[2], dictTrain,"machine") for i in machine]).astype(np.float32))
    machineY = torch.from_numpy(np.array(labelEncode(machine))).long()
    languageX = torch.from_numpy(np.array([embedding(i[2], dictTrain,"language") for i in language]).astype(np.float32))
    languageY = torch.from_numpy(np.array(labelEncode(language))).long()

    print("Extract X and Y from test set: ")
    processTestX = torch.from_numpy(np.array([embedding(i[2], dictTrain, "process") for i in processTest]).astype(np.float32))
    processTestY = torch.from_numpy(np.array(labelEncode(processTest))).long()
    machineTestX = torch.from_numpy(np.array([embedding(i[2], dictTrain, "machine") for i in machineTest]).astype(np.float32))
    machineTestY = torch.from_numpy(np.array(labelEncode(machineTest))).long()
    languageTestX = torch.from_numpy(np.array([embedding(i[2], dictTrain, "language") for i in languageTest]).astype(np.float32))
    languageTestY = torch.from_numpy(np.array(labelEncode(languageTest))).long()

    # Part 1.3
    learning_rate, epochs = 1.0, 300
    print("\nTraining Process...")
    modelProcess = LogReg(len(dictTrain)*2,processClasses)
    sgdProcess = torch.optim.SGD(modelProcess.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for i in range(epochs):
        modelProcess.train()
        sgdProcess.zero_grad()

        #forward pass:
        ypred = modelProcess(processX)
        loss = loss_func(ypred, processY)
        #backward:
        loss.backward()
        sgdProcess.step()
        if i % 20 == 0:
            print("  epoch: %d, loss: %.5f" %(i, loss.item()))

    print("\nTraining Machine...")
    modelMachine = LogReg(len(dictTrain) * 2, machineClasses)
    sgdMachine = torch.optim.SGD(modelMachine.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for i in range(epochs):
        modelMachine.train()
        sgdMachine.zero_grad()

        # forward pass:
        ypred = modelMachine(machineX)
        loss = loss_func(ypred, machineY)
        # backward:
        loss.backward()
        sgdMachine.step()
        if i % 20 == 0:
            print("  epoch: %d, loss: %.5f" % (i, loss.item()))

    print("\nTraining Language...")
    modelLanguage = LogReg(len(dictTrain) * 2, languageClasses)
    sgdLanguage = torch.optim.SGD(modelLanguage.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for i in range(epochs):
        modelLanguage.train()
        sgdLanguage.zero_grad()

        # forward pass:
        ypred = modelLanguage(languageX)
        loss = loss_func(ypred, languageY)
        # backward:
        loss.backward()
        sgdLanguage.step()
        if i % 20 == 0:
            print("  epoch: %d, loss: %.5f" % (i, loss.item()))

    print("\nPredicting Values:")
    # Part 1.4 Predict
    with torch.no_grad():
        # Process
        print("\nProcess:")
        ytestpred_prob = modelProcess(processTestX)
        ytestpred_class = ytestpred_prob.numpy().tolist()

        print("predictions for process.NOUN.000018:",ytestpred_class[3])
        print("predictions for process.NOUN.000024:", ytestpred_class[4])

        ytestpred_class = [i.index(max(i)) for i in ytestpred_class]
        processYTestList = processTestY.numpy().tolist()
        countProcess = 0
        for i in range(len(processYTestList)):
            if ytestpred_class[i] == processYTestList[i]:
                countProcess+=1
        print("process correct:",countProcess/len(processYTestList))
        print(countProcess,"out of",len(processYTestList))

        # Machine
        print("\nMachine:")
        ytestpred_prob = modelMachine(machineTestX)
        ytestpred_class = ytestpred_prob.numpy().tolist()

        print("predictions for machine.NOUN.000004:", ytestpred_class[0])
        print("predictions for machine.NOUN.000008:", ytestpred_class[1])

        ytestpred_class = [i.index(max(i)) for i in ytestpred_class]
        machineYTestList = machineTestY.numpy().tolist()
        countMachine = 0
        for i in range(len(machineYTestList)):
            if ytestpred_class[i] == machineYTestList[i]:
                countMachine += 1
        print("machine correct:", countMachine / len(machineYTestList))
        print(countMachine,"out of",len(machineYTestList))

        # Language
        print("\nLanguage:")
        ytestpred_prob = modelLanguage(languageTestX)
        ytestpred_class = ytestpred_prob.numpy().tolist()

        print("predictions for process.NOUN.000018:", ytestpred_class[1])
        print("predictions for process.NOUN.000024:", ytestpred_class[2])

        ytestpred_class = [i.index(max(i)) for i in ytestpred_class]
        languageYTestList = languageTestY.numpy().tolist()
        countLanguage = 0
        for i in range(len(languageYTestList)):
            if ytestpred_class[i] == languageYTestList[i]:
                countLanguage += 1
        print("language correct:", countLanguage / len(languageYTestList))
        print(countLanguage, "out of", len(languageYTestList))

    # Part 2.1 Co-occurrence matrix
    print("\nCalculating Matrix... will take a while")
    coMatrix = torch.from_numpy(np.array(cooccurrence(process,machine,language,dictTrain))).float()

    # Part 2.2 PCA dimension reduction
    coMatrix = (coMatrix - coMatrix.mean(dim=1,keepdims=True)[0])/coMatrix.std(dim=1,keepdims=True)[0]
    u,s,v = torch.svd(coMatrix)
    u = u.numpy().tolist()
    Udict = {}
    for i in range(len(dictTrain)):
        Udict[dictTrain[i]] = u[i][:50]
    Udict["OOV"] = u[-1][:50]

    # Part 2.3 Euclidean distance
    print("\nOutput Euclidean distance: ")
    print("('language', 'process') :",euDistance(Udict["language"],Udict["process"]))
    print("('machine', 'process') :", euDistance(Udict["machine"],Udict["process"]))
    print("('language', 'speak') :", euDistance(Udict["language"],Udict["speak"]))
    print("('word', 'words') :", euDistance(Udict["word"], Udict["words"]))
    print("('word', 'the') :", euDistance(Udict["word"], Udict["the"]))

    # Part 3.1 Redoing embedding with PCA
    processXNew = torch.from_numpy(np.array([PCAembedding(i[2], Udict, "process") for i in process]).astype(np.float32))
    processYNew = torch.from_numpy(np.array(labelEncode(process))).long()
    machineXNew = torch.from_numpy(np.array([PCAembedding(i[2], Udict, "machine") for i in machine]).astype(np.float32))
    machineYNew = torch.from_numpy(np.array(labelEncode(machine))).long()
    languageXNew = torch.from_numpy(np.array([PCAembedding(i[2], Udict, "language") for i in language]).astype(np.float32))
    languageYNew = torch.from_numpy(np.array(labelEncode(language))).long()

    processTestXNew = torch.from_numpy(np.array([PCAembedding(i[2], Udict, "process") for i in processTest]).astype(np.float32))
    processTestYNew = torch.from_numpy(np.array(labelEncode(processTest))).long()
    machineTestXNew = torch.from_numpy(np.array([PCAembedding(i[2], Udict, "machine") for i in machineTest]).astype(np.float32))
    machineTestYNew = torch.from_numpy(np.array(labelEncode(machineTest))).long()
    languageTestXNew = torch.from_numpy(np.array([PCAembedding(i[2], Udict, "language") for i in languageTest]).astype(np.float32))
    languageTestYNew = torch.from_numpy(np.array(labelEncode(languageTest))).long()

    # Part 3.2 retrain models
    learning_rate, epochs = 1.3, 350
    modelProcessNew = LogReg(200, processClasses)
    sgdProcess = torch.optim.SGD(modelProcessNew.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    print("\nTrain models by using new embedding...")
    print("\nTraining New Process...")
    for i in range(epochs):
        modelProcessNew.train()
        sgdProcess.zero_grad()

        # forward pass:
        ypred = modelProcessNew(processXNew)
        loss = loss_func(ypred, processYNew)
        # backward:
        loss.backward()
        sgdProcess.step()
        if i % 20 == 0:
            print("  epoch: %d, loss: %.5f" % (i, loss.item()))

    modelMachineNew = LogReg(200, machineClasses)
    sgdProcess = torch.optim.SGD(modelMachineNew.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    print("\nTraining New Machine...")
    for i in range(epochs):
        modelMachineNew.train()
        sgdProcess.zero_grad()

        # forward pass:
        ypred = modelMachineNew(machineXNew)
        loss = loss_func(ypred, machineYNew)
        # backward:
        loss.backward()
        sgdProcess.step()
        if i % 20 == 0:
            print("  epoch: %d, loss: %.5f" % (i, loss.item()))

    modelLanguageNew = LogReg(200, languageClasses)
    sgdProcess = torch.optim.SGD(modelLanguageNew.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    print("\nTraining New Language...")
    for i in range(epochs):
        modelLanguageNew.train()
        sgdProcess.zero_grad()

        # forward pass:
        ypred = modelLanguageNew(languageXNew)
        loss = loss_func(ypred, languageYNew)
        # backward:
        loss.backward()
        sgdProcess.step()
        if i % 20 == 0:
            print("  epoch: %d, loss: %.5f" % (i, loss.item()))

    # Part 3.3 Testing new models
    print("\nPredicting Values:")
    with torch.no_grad():
        # Process
        print("\nProcess:")
        ytestpred_prob = modelProcessNew(processTestXNew)
        ytestpred_class = ytestpred_prob.numpy().tolist()

        print("predictions for process.NOUN.000018:", ytestpred_class[3])
        print("predictions for process.NOUN.000024:", ytestpred_class[4])

        ytestpred_class = [i.index(max(i)) for i in ytestpred_class]
        processYTestList = processTestYNew.numpy().tolist()
        countProcess = 0
        for i in range(len(processYTestList)):
            if ytestpred_class[i] == processYTestList[i]:
                countProcess+=1
        print("process correct:",countProcess/len(processYTestList))
        print(countProcess,"out of",len(processYTestList))

        # Machine
        print("\nMachine:")
        ytestpred_prob = modelMachineNew(machineTestXNew)
        ytestpred_class = ytestpred_prob.numpy().tolist()

        print("predictions for machine.NOUN.000004:", ytestpred_class[0])
        print("predictions for machine.NOUN.000008:", ytestpred_class[1])

        ytestpred_class = [i.index(max(i)) for i in ytestpred_class]
        machineYTestList = machineTestYNew.numpy().tolist()
        countMachine = 0
        for i in range(len(machineYTestList)):
            if ytestpred_class[i] == machineYTestList[i]:
                countMachine += 1
        print("machine correct:", countMachine / len(machineYTestList))
        print(countMachine, "out of", len(machineYTestList))

        # Language
        print("\nLanguage:")
        ytestpred_prob = modelLanguageNew(languageTestXNew)
        ytestpred_class = ytestpred_prob.numpy().tolist()

        print("predictions for process.NOUN.000018:", ytestpred_class[1])
        print("predictions for process.NOUN.000024:", ytestpred_class[2])

        ytestpred_class = [i.index(max(i)) for i in ytestpred_class]
        languageYTestList = languageTestYNew.numpy().tolist()
        countLanguage = 0
        for i in range(len(languageYTestList)):
            if ytestpred_class[i] == languageYTestList[i]:
                countLanguage += 1
        print("process correct:", countLanguage / len(languageYTestList))
        print(countLanguage, "out of", len(languageYTestList))




