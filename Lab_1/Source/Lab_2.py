def long_substr(str):
    temp = ""
    dict = {}
    for j in range(len(str)):
        for i in range(j,len(str)):
            if not(str[i] in temp):
                temp += str[i]
            else :
                dict[temp] = len(temp)
                temp = ''
                break
    max_val = max(dict.values())
    list1=[]
    for key, val in dict.items():
        if max_val == val:
            list1.append((key, val))
    print(list1)


if __name__ == '__main__':
   long_substr("pwwkew")