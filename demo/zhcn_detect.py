
def get_zhcn_number(char):
    """
    判断字符串中，中文的个数
    :param char: 字符串
    :return:
    """
    count = 0
    for item in char:
        if 0x4E00 <= ord(item) <= 0x9FA5:
            count += 1
    return count

s = "我说china"
print(s, len(s), get_zhcn_number(s))

s1 = "我说：hk？"
print(s1, len(s1), get_zhcn_number(s1))

s2 = "我说：hk?"
print(s2, len(s2), get_zhcn_number(s2))

s3 = "我是"
print(s3.isalpha())