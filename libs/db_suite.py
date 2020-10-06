import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, MetaData, ForeignKey, select, exists
from cryptography.hazmat.primitives import hashes
import cryptography
import numpy as np
# please run, if you want to use a local poastgresdb: python3 -m pip install python-psycopg2
# You may have also to install the other imported libaries
# local database which one have to create see https://www.postgresqltutorial.com/install-postgresql/
# "postgres+psycopg2://[postgres, actually user, per default postgres]:[Password]@[127.0.0.1, host here local]/[Name of db]"
engine = create_engine("postgres+psycopg2://postgres:+da_Quantum235!@127.0.0.1/qdb")

def init_db(engine):
    """
    Creates the main table, please just call once.
    Parameter:
        - engine, sql-engine
    """
    metadata = MetaData()
    main = Table('main', metadata,
        Column('hash_val', String, primary_key=True), Column('Params_and_description', String))
    metadata.create_all(engine)

translater = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l',\
'm','n','o','p','q','r','s','t','u','v']
def to_32(bin_str):
    """
    Returns a 32 system string for a binary string
    """
    out = ''
    while len(bin_str) >= 5:
        block = bin_str[-5:]
        bin_str = bin_str[:-5]
        out = translater[int(block, 2)] + out
    if len(bin_str) != 0:
        out = translater[int(bin_str, 2)] + out
    return out 

def to_str_hash(string):
    """
    Returns for an arbritary string the hash value
    """
    digest = hashes.Hash(hashes.SHA256(),cryptography.hazmat.backends.default_backend())
    digest.update(bytes(string, 'utf-8'))
    return to_32(str(bin(int(str(digest.finalize().hex()),16)))[2:])

def get_par_str(descr, params):
    """
    Maps description and Parameter to string
    Parameter:
        - descr, string
        - params, dict
    """
    assert isinstance(params, dict), "params has to be a dictionary."
    out = descr + '\n'
    sorted_keys = sorted(params.keys())
    for key in sorted_keys[:-1]:
        out += key + ':' + str(params[key]) + ', '
    key = sorted_keys[-1]
    out += key + ':' + str(params[key])
    return out 

def write_db(engine, descr, params, tables, tag):
    """
    Creates or appends new data to the database.

    Parameter:
        - engine, sql-engine
        - descr, string
        - params, dict
        - tables, dict
    Return
        success value, bool
    """
    assert not 'tag' in tables, "'tag' is a forbitten key in tables. Please rename this key"
    conn = engine.connect()
    para_string = get_par_str(descr, params)
    hash_val = to_str_hash(para_string)
    metadata = MetaData()
    main_tab = Table('main', metadata, autoload=True, autoload_with=engine)
    entry = select([main_tab.c.hash_val]).where(main_tab.c.hash_val == hash_val)
    ex_result = len(conn.execute(entry).fetchall()) != 0
    if not ex_result:
        ins = main_tab.insert().values(hash_val = hash_val, Params_and_description = para_string)
        ins.compile()
        conn.execute(ins)
    else:
        hash_tab = Table(hash_val, metadata, autoload=True, autoload_with=engine)
        entry = select([hash_tab.c.tag]).where(hash_tab.c.tag == tag)
        ex_tag = len(conn.execute(entry).fetchall()) != 0
        if ex_tag:
            print("There is allready data with this tag. Please make sure that you don't save the same data twice and eventually change the tag.")
            return False
    n = len(tables[list(tables.keys())[0]])
    tables['tag'] = [tag]*n
    df = pd.DataFrame(tables)
    df.to_sql(hash_val, conn, if_exists = 'append')
    return True

def read_db(engine, descr, params):
    """
    Reads data from the database.

    Parameter:
        - engine, sql-engine
        - descr, string
        - params, dict
    Return
        - False, if the process fails
        - pandas Dataframe with the data otherwise
    """
    conn = engine.connect()
    para_string = get_par_str(descr, params)
    hash_val = to_str_hash(para_string)
    metadata = MetaData()
    main_tab = Table('main', metadata, autoload=True, autoload_with=engine)
    entry = select([main_tab.c.hash_val]).where(main_tab.c.hash_val == hash_val)
    ex_result = len(conn.execute(entry).fetchall()) != 0
    if not ex_result:
        print('No entry for this input')
        return False
    return pd.read_sql(hash_val, conn, index_col="index")

def delete_table(engine, **kwargs):
    """
    Deletes data from the database.

    Parameter:
        - engine, sql-engine
        
        - descr, string
        - params, dict
        or
        - hash_val
    Return
        - False, if the process fails
    """
    conn = engine.connect()
    if not 'hash_val' in kwargs:
        assert 'descr' in kwargs and 'params' in kwargs, "delete_table needs keyword arguments descr and params or hash_val"
        para_string = get_par_str(kwargs['descr'], kwargs['params'])
        hash_val = to_str_hash(para_string)
    else: 
        hash_val = kwargs['hash_val']
    metadata = MetaData()
    main_tab = Table('main', metadata, autoload=True, autoload_with=engine)
    entry = select([main_tab.c.hash_val]).where(main_tab.c.hash_val == hash_val)
    ex_result = len(conn.execute(entry).fetchall()) != 0
    if not ex_result:
        print('unkown entry')
        return False
    conn.execute(main_tab.delete().where(main_tab.c.hash_val == hash_val))
    metadata2 = MetaData()
    hash_tab = Table(hash_val, metadata2, autoload=True, autoload_with=engine)
    hash_tab.drop(engine)
    return True

def show_all(engine):
    """Shows all saved parameter

    - engine, sql-engine
    """
    conn = engine.connect()
    main_tab = pd.read_sql('main', conn)
    return main_tab

    
desc = "Test setup"
params = {"Date": "28.09.2020", "Name": "Fabian"}
tables = {"nums":[1,2,3,4,5,6,7],"prim":[0,1,1,0,1,0,1]}
#init_db()
write_db(engine, desc, params, tables,0)
#delete_table(descr=desc, params = params)
print(show_all(engine, ))
print(read_db(engine, desc,params))
