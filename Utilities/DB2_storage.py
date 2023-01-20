from os import environ
import ibm_db_sa
import sqlalchemy.engine
from sqlalchemy import create_engine
import numpy as np


class IbmConnection:
    def __init__(self, table):
        self.__table = table
        self.__username = environ['username']
        self.__password = environ['password']
        self.__hostname = environ['hostname']
        self.__port = environ['port']
        self.__database = environ['database']
        self.__security = environ['security']
        self.__connection: sqlalchemy.engine.Engine | None = None
        self.__column_size: int = 2 * (int(self.__table.split('_')[-1]) + 2)

    def connect(self):
        url = (
            f"ibm_db_sa://{self.__username}:{self.__password}@{self.__hostname}:{self.__port}/{self.__database}?"
            f"security={self.__security}")
        self.__connection = create_engine(url)
        self.__connection.connect()

    @property
    def is_connected(self) -> bool:
        return self.__connection is not None

    @property
    def server(self):
        return ibm_db_sa.server_info(self.__connection) if self.__connection else None

    @property
    def client(self):
        return ibm_db_sa.client_info(self.__connection) if self.__connection else None

    def close(self):
        if self.__connection:
            self.__connection.dispose()
        self.__connection = None

    def __str__(self):
        return f'{self.__username}.{self.__table}'

    def create_table(self):
        if not self.__connection:
            raise ConnectionRefusedError('Not Connected')
        q = ','.join([f'U{i} float not null, V{i} float not null' for i in range((self.__column_size - 1) // 2)])
        query = f'Create table {self.__username}.{self.__table} (U float not null, V float not null, {q});'
        self.__connection.execute(query)
        print(f'{self.__username}.{self.__table} created')

    def insert(self, df: np.ndarray):
        if not self.__connection:
            raise ConnectionRefusedError('Not Connected')
        assert df.shape[-1] == self.__column_size, 'Data size does not match'
        values = ',\n'.join([f'({",".join(map(str, n))})' for n in df])
        statement = f'Insert into {self.__username}.{self.__table} \nvalues {values};'
        self.__connection.execute(statement)

    def extract(self):
        if not self.__connection:
            raise ConnectionRefusedError('Not Connected')
        statement = f"select * from {self.__username}.{self.__table};"
        results = self.__connection.execute(statement)
        return np.asarray(results.fetchall())

    def drop_table(self):
        if not self.__connection:
            raise ConnectionRefusedError('Not Connected')
        statement = f'drop table {self.__username}.{self.__table};'
        try:
            self.__connection.execute(statement)
        except:
            pass
