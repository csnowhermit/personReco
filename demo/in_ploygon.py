import os

'''
    判断某点是否在多边形里面
    射线法：以判断点开始，向右（或向左）水平方向做一射线，计算该射线与多边形的每条边的交点为u个数，
        如果交点为奇数个，则该点位于多边形内；
        如果为偶数个，则位于多边形外；
'''

'''
    pt点是否在poly多边形内
    :return 在多边形内，True；否则，False
'''
def isInsidePolygon(pt, poly):
    c = False
    i = -1
    l = len(poly)
    j = l - 1
    while i < l - 1:
        i += 1
        # print(i, poly[i], j, poly[j])    # 多边形各点
        if ((poly[i]["x"] <= pt["x"] and pt["x"] < poly[j]["x"]) or (
                poly[j]["x"] <= pt["x"] and pt["x"] < poly[i]["x"])):
            if (pt["y"] < (poly[j]["y"] - poly[i]["y"]) * (pt["x"] - poly[i]["x"]) / (
                    poly[j]["x"] - poly[i]["x"]) + poly[i]["y"]):
                c = not c
        j = i
    return c


if __name__ == '__main__':
    # print(isPointinPolygon([0.8, 0.9], [[0, 0], [1, 1], [0, 1], [0, 0]]))
    # abc = [{'x': 1, 'y': 1}, {'x': 1, 'y': 4}, {'x': 3, 'y': 7}, {'x': 4, 'y': 4}, {'x': 4, 'y': 1}]
    # print(isInsidePolygon({'x': 2, 'y': 2}, abc))
    polyList = [{'x': 200, 'y': 180}, {'x': 1000, 'y': 180}, {'x': 0, 'y': 720}, {'x': 1000, 'y': 720}]
    personList = [[{'x': 685, 'y': 139}, {'x': 839, 'y': 139}, {'x': 685, 'y': 397}, {'x': 839, 'y': 397}],
                  [{'x': 347, 'y': 186}, {'x': 504, 'y': 186}, {'x': 347, 'y': 378}, {'x': 504, 'y': 378}],
                  [{'x': 573, 'y': 34}, {'x': 612, 'y': 34}, {'x': 573, 'y': 171}, {'x': 612, 'y': 171}],
                  [{'x': 355, 'y': 64}, {'x': 405, 'y': 64}, {'x': 355, 'y': 202}, {'x': 405, 'y': 202}],
                  [{'x': 646, 'y': 49}, {'x': 685, 'y': 49}, {'x': 646, 'y': 136}, {'x': 685, 'y': 136}],
                  [{'x': 629, 'y': 7}, {'x': 650, 'y': 7}, {'x': 629, 'y': 61}, {'x': 650, 'y': 61}],
                  [{'x': 552, 'y': 41}, {'x': 591, 'y': 41}, {'x': 552, 'y': 173}, {'x': 591, 'y': 173}]
                 ]

    for person in personList:
        for point in person:
            if isInsidePolygon(point, polyList) is True:
                print(person)


