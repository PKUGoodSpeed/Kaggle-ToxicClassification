
class TrainedModel(object):
    ''' each trained model needs to implement predict method
    '''

    def __init__(self, md):
        self.md = md

    def predict(self, test, **kwargs):
        raise ValueError("Needs to be implemented")

    def save(self, mDB, nameKey):
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



