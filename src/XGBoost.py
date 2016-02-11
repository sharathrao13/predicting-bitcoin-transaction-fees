param = {'bst:max_depth':10, 'bst:eta':1, 'silent':1, 'objective':'reg:linear'}
    param['nthread'] = 4
    param['eval_metric'] = 'rmse'
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    num_round = 200
    bst = xgb.train( param.items(), dtrain, num_round, evallist )
    bst.save_model('0001.model')
    ypred = bst.predict(dtest)

    print(str(ypred.shape))
    print(str(dtest.num_col()))
    print(str(dtest.num_row()))
    errors = []
    for i in range(0, len(ypred)):
        if ds_outputs[i] > 0:
            errors.append((abs(round(ypred[i])- ds_outputs[i]) / ds_outputs[i]) * 100)
        if i < 10000:
            print('Pred: '+str(ypred[i])+' Actual: '+str(ds_outputs[i]))
    print('Error % '+str(np.mean(errors)))
    print("--- %s seconds ---" % (time.time() - start))
if __name__ == "__main__":
    xgboost()
