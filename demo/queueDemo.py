from utils.capUtil import Stack

mq = Stack(3)

# for i in range(10):
#     mq.push(str(i) + "==")

prefix = "mp4"
for i in range(100):
    s = "%s_%04d" % (prefix, i)
    print(s)