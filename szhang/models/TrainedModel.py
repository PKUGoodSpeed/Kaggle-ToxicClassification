import os,sys
sys.path.append('..')
import modelDB

class TrainedModel(object):
    ''' each trained model needs to implement predict method
    '''

    def __init__(self, md = None):
        self.md = md

    def setModel(self, md):
        if self.md is not None:
            raise ValueError("md already set")

        self.md = md

    def predict(self, test, **kwargs):
        raise ValueError("Needs to be implemented")

    def save(self, mDB, nameKey, modelpath, **kwargs):
        ''' save model into modelpath/nameKey,
            update the information into modelDB
            :mDB: meta data frame storing all models info
            :nameKey: unique identifier for each saved model
            :modelpath: subdir inside modelDB dir, e.g. if modelDB is /root/modelDB, then modelpath is /cnn
        '''

        #house keeping
        absPath = os.path.join(modelDB.MODEL_DB_ROOT, modelpath)
        if not os.path.exists( absPath ):
            print("Creating model path {}".format(absPath))
            os.mkdir(absPath)

        #check for nameKey conflict
        #schema of db
        # 'modelName', type {rnn, cnn, rf}, date, model
        if (not mDB.empty) and any( mDB['modelName'] == nameKey ):
            delete = raw_input("Warning: found identical nameKey, overwrite: [y/n]?")
            if delete != 'y' and delete != 'Y':
                print("aborting")
                return mDB
            else:
                mDB = mDB[mDB['modelName'] != nameKey]

        mDB = self._save(mDB, nameKey, modelpath, **kwargs)
        return mDB

    def _save(self, mDB, nameKey, modelpath, **kwargs):
        '''
            to be implemented by subclass
        '''
        raise ValueError("Needs to be implemented")

    def load(self, mDB, nameKey, modelpath):
        raise ValueError("Needs to be implemented")


class TrainedModelEnsemble(object):
    ''' use each trained model to predict
        combine to obtain final resutls
    '''

    def load_model(self, mDB):
        ''' load selected trained model
        '''
        pass


    def predict_ensemble(self):
        '''
        run prediction for each model
        combine using ensemble strategy
        '''

        pass



