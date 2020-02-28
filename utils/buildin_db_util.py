
import sqlite3
import pandas as pd

'''
    内置化数据库操作
'''

conn = sqlite3.connect('../data/person-search.db')
cursor = conn.cursor()

'''
    表是否存在
'''
def isExists(tablename):
    sql = '''
        select * from sqlite_master where type = 'table' and name = '%s'
    ''' % tablename
    cursor.execute(sql)
    result = cursor.fetchall()
    if len(result) > 0:
        return True
    else:
        return False



def createPersonInfoTab():
    sql = '''
        create table person_infos(
            personID TEXT, 
            personName TEXT, 
            personType TEXT, 
            markColor TEXT
        )
    '''
    cursor.execute(sql)

def saveData2Table():
    values = [['1', '张三', '站务', '[40, 92, 230]'],
              ['2', '李四', '保洁', '[176, 97, 248]']]

    for v in values:
        sql = '''
            insert into person_infos(personID, personName, personType, markColor) 
            values (%s, %s, %s, %s)
        ''' % (v[0], v[1], v[2], v[3])
        cursor.execute(sql)
        cursor.execute("commit")

def getPersonInfos():
    sql = '''
        select personID, personName, personType, markColor 
        from person_infos
    '''
    cursor.execute(sql)
    result = cursor.fetchall()
    for r in result:
        print(r)


if __name__ == '__main__':
    # print(conn)
    saveData2Table()

    if isExists("person_infos") is False:
        createPersonInfoTab()
        saveData2Table()
    getPersonInfos()

    values = [['1', '张三', '站务', '[40, 92, 230]'],
              ['2', '李四', '保洁', '[176, 97, 248]']]
    for v in values:
        print(v[0], v[1], v[2], v[3])

