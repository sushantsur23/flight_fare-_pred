from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from Logger import Logger

logging = Logger('logfile.log')


class Connector:
    def Connection(self):
        cloud_config = {'secure_connect_bundle': 'E:\Ineuron\DS\Internship\Flight-Fare-Prediction-main\secure-connect-ineuron .zip'}
        clientid = 'YmAUlbZOJYTmbYefCJzwabci'
        Client_secret = 'rnX9uQxQXEWEQZ0qL6Gt6jkz7edh4uwdPYCg0QQKzFJYlR_.K0Yi-wy99,.JKUSsiFrCr,Jt.ehYCS3BOYpqZ2IdvwH7di1bQarrMRenniq-0GYMLe4hnJrjt0Cmrurl'
        auth_provider = PlainTextAuthProvider(clientid, Client_secret)
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        self.session = cluster.connect('flight')
        self.session.execute("CREATE TABLE faredata17(id uuid PRIMARY KEY,Airline text,Source text,Destination text,Total_Stops int,Departure int,Output int);")
        print("Created Successfully")
    def addData(self, result):
        logging.log_operation('INFO', "Inside addData")
        logging.log_operation('INFO', "Inside addData")

        column = "id, Airline, Source,Destination, Total_Stops, Total_Duration, Journey_month, Journey_day"
        value = "{0},'{1}','{2}','{3}',{4},{5},{6}".format('uuid()', result['airline'], result['Source'],
                                                                     result['Destination'], result['Total_Stops'],
                                                                    result['date_dep'],result['output'])
        logging.log_operation('INFO', "String created")
        custom = "INSERT INTO faredata17({}) VALUES({});".format(column, value)

        logging.log_operation('INFO', "Key created")
        self.session.execute("USE flight")

        output = self.session.execute(custom)

        logging.log_operation('INFO', "Column inserted {}".format(output))

    def getData(self):
        self.session.execute("use flight")
        row = self.session.execute("SELECT * FROM faredata17;")
        collection = []
        for i in row:
            collection.append(tuple(i))
            logging.log_operation('INFO', "Retrieved Data from Database : {}".format(i))
            return tuple(collection)
