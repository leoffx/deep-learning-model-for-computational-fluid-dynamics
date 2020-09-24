model.fit(x=[x_train[:-5], obj_train[:-5]], y=[x_train[i:i-5]
                                               for i in range(5)], batch_size=4, epochs=5, callbacks=[vis, sav])
