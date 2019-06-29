#==============================================================================
#  Loading and saving datasets
#==============================================================================

import numpy as np
#np.set_printoptions(threshold=np.inf) #View full arrays in console
import pickle
from random import shuffle


data_folder = './'


def load_any_data(filename):
    '''
    General case to load any data from filename
    Can be stored in .npz or .pkl or .pkl.gz form
    Output tuple: xtr,ytr,xva,yva,xte,yte
    '''
    loaded = np.load(filename)
    xtr = loaded['xtr']
    ytr = loaded['ytr']
    xva = loaded['xva']
    yva = loaded['yva']
    xte = loaded['xte']
    yte = loaded['yte']
    return (xtr, ytr, xva, yva, xte, yte)




#==============================================================================
# Reuters RCV1-v2
#==============================================================================
'''
WORKFLOW:
    Download relevant files from http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm
        Online Appendix 2 - Topics
        Online Appendix 8 - Articles IDs with topics
        Online Appendix 12 - Article IDs with tokens
    Rename OA8 to article_categories_all.dat
    Delete all categories except 2nd level in OA2 and call it categories_level2.txt
    Run organize_data() to get data.dat
    Run build_data(cat_filename = 'categories_level2', build_data_filename = 'data_level2') to get data with only level2 articles. Takes a while
    Run label_frequency(cat_filename = 'categories_level2', data_filename = 'data_level2')
        This gives some very frequent categories and some very infrequent categories (see stuff.xlsx)
        I deleted C15, C42, GOBIT, GVOTE, GMIL from categories_level2 and saved in categories_level2_final50
    Run build_data(cat_filename = 'categories_level2_final50', build_data_filename = 'data_level2_final50') to get data with only relevant articles. Takes a while
    Run label_frequency(cat_filename = 'categories_level2_final50', data_filename = 'data_level2_final50')
        This gives a DIFFERENT distribution (see stuff.xlsx)
        Since some applicable cats have been deleted, some articles which were previously ineligible due to having 2 applicable cats have now become eligible
        So the article count for all remaining eligible cats will go up
        Eg: Article 50959 had C15 and GTOUR, but with C15 being deleted, it only has GTOUR, so becomes eligible and increases the count for GTOUR
    Run most_common_tokens(numtokens=2000, data_filename = 'data_level2_final50')
        Saves the max occurring tokens with counts as a list of tuples in a pkl file
        Max 2000 occurring tokens with counts are also saved in stuff.xlsx
        If less tokens are to be considered (say 400), just load the pkl file, extract the 1st 400 elements, and save that as a new pkl file
    Run build_mldata(numval=50000, numtest=100000, numtokens=xxx, data_filename = 'data_level2_final50') to get final npz data
        xxx can be 2000 or 400, etc
'''

def organize_data():
    '''
    Combine data from the 5 original files in Online Appendix 12 (http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm)
    into a single file tokens.dat
    Delete .W and blank lines
    Combine all tokens in 1 article into a single line
    Every entry in tokens.dat has:
        1st line: .I <article ID>
        2nd line: <list of tokens>
    '''
    with open(data_folder + 'dataset_RCV1/data.dat','w') as fw:
        for ftbr in ['train','test_pt0','test_pt1','test_pt2','test_pt3']:
            with open(data_folder + 'dataset_RCV1/lyrl2004_tokens_{0}.dat'.format(ftbr),'r') as fr:
                for line in fr:
                    if line[:2]=='.W':
                        continue
                    elif line[:2]=='.I':
                        fw.write(line)
                        newline = '' #start a new line which will have all tokens in the article
                    elif line=='\n':
                        fw.write(newline + '\n') #end the new line for the article
                    else:
                        newline += line[:-1] + ' ' #add tokens to new line


def extract_categories(filename = 'categories_level2_final50'):
    '''
    Extracts categories from filename and returns them as a list
    '''
    with open(data_folder + 'dataset_RCV1/{0}.txt'.format(filename),'r') as f:
        catlines = f.readlines()
    numcats = len(catlines)
    cats = numcats*[''] #list to hold all categories
    for i in range(numcats):
        for char in catlines[i][23:]: #this is where the category name starts from
            if char==' ': #category name ends
                break
            else: #category name continues
                cats[i] += char
    return cats


def build_data(cat_filename = 'categories_level2_final50', build_data_filename = 'data_level2_final50'):
    '''
    Get data from data.dat and categories from cat_filename
    Delete all articles which don't have correct categories
    Save in build_data_filename a new version of tokens which only has articles with correct category
    '''
    cats = set(extract_categories(filename = cat_filename))
    with open(data_folder + 'dataset_RCV1/data.dat','r') as frtok:
        tokenlines = frtok.readlines()
    idlines = tokenlines[0:len(tokenlines):2] #only hold article ID lines
    with open(data_folder + 'dataset_RCV1/article_categories_all.dat','r') as frcat:
        artcatlines = frcat.readlines()
    current_cats = set() #hold categories of current article
    current_art = '2286' #1st article id
    fullindex = 0
    with open(data_folder + 'dataset_RCV1/{0}.dat'.format(build_data_filename),'w') as fw:
        for i in range(len(artcatlines)):
            if i%10000==0: print(i) #progress
            previous_art = current_art
            cat, current_art, _ = artcatlines[i].split(' ') #category, article id, 1
            if previous_art == current_art: #still in same article
                current_cats.add(cat)
            else: #on to next article
                num_correct_cats = cats.intersection(current_cats) #see how many categories are correct
                current_cats = set((cat,)) #refresh current_cats and add category of new article (current_art) to it
                if len(num_correct_cats)==1: #if there's only 1 correct category
                    try:
                        index = idlines.index('.I {0}\n'.format(previous_art)) #find line number of the applicable article
                    except ValueError:
                        continue
                    fullindex += index #fullindex keeps track of absolute index (index is relative index since idlines is getting truncated every time)
                    fw.write(tokenlines[fullindex*2][:-1] + ' ' + num_correct_cats.pop() + ' \n') #write ".I <article ID> <article category> "
                    fw.write(tokenlines[fullindex*2+1]) #write the tokens
                    del idlines[:index] #make future searches simpler, since articles are in increasing order. This will make idlines start at the article which just got added


def label_frequency(cat_filename = 'categories_level2_final50', data_filename = 'data_level2_final50'):
    '''
    Find number of articles in data_filename in which each category from cat_filename occurs
    '''
    cats = extract_categories(filename = cat_filename)
    catfreqs = {key:value for (value,key) in enumerate(cats)} #create dictionary where each key is an element in cats and the corresponding value will hold its frequency
    with open(data_folder + 'dataset_RCV1/{0}.dat'.format(data_filename),'r') as f:
        tokenlines = f.readlines()
    idlines = tokenlines[0:len(tokenlines):2] #only lines with article IDs
    ids = ''.join(idlines) #combine into single string
    for key in catfreqs:
        catfreqs[key] = ids.count(' '+key+' \n')
    catfreqs_sort = [(k,catfreqs[k]) for k in sorted(catfreqs, key=catfreqs.get, reverse=True)] #sort according to dict values, i.e. highest frequency first
    return catfreqs_sort


def most_common_tokens(numtokens=2000, data_filename = 'data_level2_final50'):
    '''
    Find how many articles each token occurs in, in data_filename
    Save 'numtokens' tokens with the highest counts as a list of tuples
        Note that this counts how many unique articles a token is present in, not the total count of a token
        For example, 'brazil' can occur 10 times each in 5 articles, but it will have lower rank than 'india' which occurs 1 time each in 6 articles
    '''
    tokendict = {}
    linecounter = 0
    with open(data_folder + 'dataset_RCV1/{0}.dat'.format(data_filename),'r') as f:
        for line in f:
            if line[:2]=='.I' or line=='\n': #don't consider ID lines
                continue
            else: #only consider token lines
                linecounter += 1
                line = line[:-1] #remove newline character at end
                for token in np.unique(line.split(' ')): #get unique tokens in article
                    if token in tokendict:
                        tokendict[token] += 1
                    else:
                        tokendict[token] = 1
                if linecounter%100000==0: print(linecounter) #progress
    tokendict.pop('') #this token gets added by default to all articles
    maxtokens_withcounts = [(k,tokendict[k]) for k in sorted(tokendict, key=tokendict.get, reverse=True)][:numtokens] #sort tokendict dictionary according to values, take the highest 'numtokens' values, return as a list of tuples
#    maxtokens = maxtokens_withcounts.keys() #these are the most commonly occurring tokens
    with open(data_folder + 'dataset_RCV1/maxtokens_withcounts_{0}.pkl'.format(numtokens),'w') as f:
        pickle.dump(maxtokens_withcounts, f) #use this file in future to skip the long processing time of this method


def build_mldata(numval=50000, numtest=100000, numtokens=2000, data_filename = 'data_level2_final50'):
    '''
    Build actual data from data_filename using maxtokens_filename as numpy arrays
    numval, numtest: How many samples to have in each
    maxtokens_to_consider: No. of features
    '''
    cats = extract_categories()
    dcats = {key:value for (value,key) in enumerate(cats)} #create dictionary where each key is an element in list and the corresponding value is its index
    numcats = len(cats)
    del cats
    with open(data_folder + 'dataset_RCV1/maxtokens_withcounts_{0}.pkl'.format(numtokens),'r') as f:
        alltokens_withcounts = pickle.load(f) #list of tuples
    alltokens = [at[0] for at in alltokens_withcounts] #only get token names, ignore counts
    shuffle(alltokens) #break up the order of the most frequent token being at pos0, next at pos1, and so on. In-place shuffle
    dalltokens = {key:value for (value,key) in enumerate(alltokens)} #create dictionary where each key is an element in list and the corresponding value is its index
    del alltokens
    with open(data_folder + 'dataset_RCV1/{0}.dat'.format(data_filename),'r') as f:
        numarticles = len(f.readlines())//2
        xdata = np.zeros((numarticles,numtokens),dtype='float32') #samples, features
        ydata = np.zeros((numarticles,numcats),dtype='uint8') #samples, categories
        article_counter = 0
    with open(data_folder + 'dataset_RCV1/{0}.dat'.format(data_filename),'r') as f: #for some reason, the file needs to be opened again
        for line in f:
            if line[:2]=='.I':
                _,_,cat,_ = line.split(' ') #.I, id, category, \n
                ydata[article_counter,dcats[cat]] = 1
            else:
                tokens = line.split(' ')
                for token in tokens:
                    try:
                        xdata[article_counter,dalltokens[token]] += 1.
                    except KeyError:
                        continue
                article_counter += 1
                if article_counter%10000==0: print(article_counter) #progress
    xdata = np.log10(1+xdata) #applying log transformation as in Hinton 2012 - 'Improving neural networks by preventing co-adaptation of feature detectors'
    '''
    Note that the numtokens and numcats axes are in no order which influences ML algorithms
        For example, numcats is in alphabetical order, but the frequency of cats are independent of alphabetical order
        numtokens is in purely random order due to shuffling
    But numarticles is in time order:
        According to the original paper, the 1st few samples (earliest in time) should be used for tr, and the remaining (later in time) for te
        To keep that, use:
            xtr = xdata[:-(numval+numtest)]
            xva = xdata[-numval:-numtest]
            xte = xdata[-numtest:]
            and likewise for ytr,yva,yte
        Instead I'll divide the data randomly (not chronologically), so I'll shuffle along the numarticles axis and then split
    '''
    shuff = np.random.permutation(numarticles)
    xdata, ydata = xdata[shuff], ydata[shuff]
    xte = xdata[:numtest]
    yte = ydata[:numtest]
    xva = xdata[numtest:numtest+numval]
    yva = ydata[numtest:numtest+numval]
    xtr = xdata[numtest+numval:]
    ytr = ydata[numtest+numval:]
    np.savez_compressed(data_folder + 'dataset_RCV1/rcv1_{0}.npz'.format(numtokens), xtr=xtr,ytr=ytr, xva=xva,yva=yva, xte=xte,yte=yte)


