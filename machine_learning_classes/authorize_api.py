from apiclient import discovery
from httplib2 import Http
from oauth2client import client, file, tools



class authorize_api(object):
    def __init__(self, conf):
        self.credentials_file_path=conf.credentials_file_path
        self.clientsecret_file_path=conf.clientsecret_file_path,
        self.scope=conf.scope



    def gdrive_authorize(self):
        # define store
        store = file.Storage(self.credentials_file_path)
        credentials = store.get()
    # get access token
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(self.clientsecret_file_path, self.scope)
            credentials = tools.run_flow(flow, store)

    # define API service
        http = credentials.authorize(Http())
        drive_api_service = discovery.build('drive', 'v3', http=http)

        return drive_api_service

