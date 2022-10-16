import decimal
import os

import boto3
from boto3.dynamodb.conditions import Attr


class AWSConnect:
    def __init__(self, table_name):
        self.__session = self.__connect()
        dynamodb_resource = self.__session.resource('dynamodb')
        self.__table = dynamodb_resource.Table(table_name)

    def insert(self, entry):
        return self.__table.put_item(entry)

    def bulk_insert(self, *entries):
        with self.__table.batch_writer() as batch:
            for entry in entries:
                batch.put_item(Item=entry)

    def extract(self, annoa_size: int, theta):
        return self.__table.scan(FilterExpression=Attr('Theta/Distribution').eq(theta) & Attr('Size').eq(annoa_size))

    def scan(self):
        return self.__table.scan()['Items']

    @staticmethod
    def __connect():
        return boto3.Session(
            aws_access_key_id=os.environ['AccessKeyId'],
            aws_secret_access_key=os.environ['SecretAccessKey'],
            region_name=os.environ['Region']
        )

    def create_table(self, table_name):
        params = {
            'TableName': table_name,
            'KeySchema': [
                {'AttributeName': 'Theta/Distribution', 'KeyType': 'HASH'},
                {'AttributeName': 'Index', 'KeyType': 'RANGE'}
            ],
            'AttributeDefinitions': [
                {'AttributeName': 'Theta/Distribution', 'AttributeType': 'S'},
                {'AttributeName': 'Index', 'AttributeType': 'N'}
            ],
            'ProvisionedThroughput': {
                'ReadCapacityUnits': 1000,
                'WriteCapacityUnits': 1000
            }
        }
        table = self.__session.resource('dynamodb').create_table(**params)
        print(f"Creating {table_name}...")
        table.wait_until_exists()
        return table
