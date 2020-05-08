import pandas as pd
import io
from googleapiclient.http import MediaIoBaseDownload
import machine_learning_projects.common_machine_learning_classes.authorize_api as auth
import sys
import os


class collect_data(object):
     def __init__(self,conf):
        self.conf=conf


     def get_raw_file(self, file_name):
        file_id=self.conf.raw_file_id
        data = self.download_file(file_id ,file_name)
        return data

     def get_test_file(self, file_name):
        file_id = self.conf.test_file_id
        data = self.download_file(file_id,file_name)
        return data

     def get_train_test_files(self,train_file_name,test_file_name):
        train_file_id=self.conf.train_file_id
        train_data=self.download_file(self,train_file_id,train_file_name)
        test_file_id=self.conf.test_file_id
        test_data = self.download_file(self,test_file_id,test_file_name)

        return train_data,test_data


     def download_file(self,file_id,file_name):
        full_file_path=os.path.abspath(self.conf.file_path+'cache'+file_name)
        df = self.read_cache_data(full_file_path)
        if df.empty:
            gdrive = auth.authorize_api(self.conf)
            drive_api_service = gdrive.gdrive_authorize()
            request = drive_api_service.files().get_media(fileId=file_id)
            # print(request.to_json())
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            try:
                while done is False:
                    status, done = downloader.next_chunk()
                    # print("Download %d%%." % int(status.progress() * 100))
                with io.open(full_file_path, 'wb') as f:
                    fh.seek(0)
                    f.write(fh.read())
            except:
                print("Unexpected error:", sys.exc_info()[0])
            df = pd.read_csv(full_file_path)
        return df

     @staticmethod
     def read_cache_data(full_filepath):

        cache_raw_data = False
        data = pd.DataFrame()

        try:
            data = pd.read_csv(full_filepath)
            cache_data = True

            print("Loading cached data: {}".format(data.shape))
        except:
            pass
        return data

     '''def wf(TICKER, n_samples, test_percentage, anchored=False):
         data = prepare_data(TICKER)
         lenght = len(data)
         test_lenght = round(n_samples * (test_percentage / 100))
         steps = round((lenght - n_samples) / test_lenght)
         print(steps)

         y_test_total = np.zeros(shape=(0, 3))

         for i in range(0, steps):
             print(i)
             if anchored == False:
                 if i == steps - 1:

                     start = ((i) * test_lenght)
                     data_wf = data[(((i) * test_lenght)): lenght]
                     y_test = build_model(data_wf, len(data_wf) - round(n_samples * (1 - (test_percentage / 100))))
                     y_test_total = np.concatenate([y_test_total, y_test])
                 else:

                     data_wf = data.iloc[(i * test_lenght):((i * test_lenght) + n_samples)]
                     end = ((i * test_lenght) + n_samples)
                     start = (i * test_lenght)
                     y_test = build_model(data_wf, test_percentage / 100)
                     y_test_total = np.concatenate([y_test_total, y_test])

             else:
                 if i == steps - 1:

                     start = 0
                     data_wf = data[0: lenght]
                     y_test = build_model(data_wf, len(data) - n_samples - (test_lenght * (i - 1)))
                     y_test_total = np.concatenate([y_test_total, y_test])


                 else:

                     data_wf = data.iloc[0:((i * test_lenght) + n_samples)]
                     end = ((i * test_lenght) + n_samples)
                     start = (0)
                     y_test = build_model(data_wf, test_lenght)
                     y_test_total = np.concatenate([y_test_total, y_test])

         alldf = np.vstack(y_test_total)

         return alldf'''