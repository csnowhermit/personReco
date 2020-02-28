
import os

'''
    范围判断工具
'''

'''
    pt角是否在poly多边形内
    :return 在多边形内，True；否则，False
'''
def isInsidePolygon(pt, poly):
    c = False
    i = -1
    l = len(poly)
    j = l - 1
    while i < l - 1:
        i += 1
        # print(i, poly[i], j, poly[j])
        if ((poly[i]["x"] <= pt["x"] and pt["x"] < poly[j]["x"]) or (
                poly[j]["x"] <= pt["x"] and pt["x"] < poly[i]["x"])):
            if (pt["y"] < (poly[j]["y"] - poly[i]["y"]) * (pt["x"] - poly[i]["x"]) / (
                    poly[j]["x"] - poly[i]["x"]) + poly[i]["y"]):
                c = not c
        j = i
    return c

if __name__ == '__main__':
    poly = [{'x': 1, 'y': 1}, {'x': 1, 'y': 4}, {'x': 3, 'y': 7}, {'x': 4, 'y': 4}, {'x': 4, 'y': 1}]
    print(isInsidePolygon({'x': 2, 'y': 2}, poly))