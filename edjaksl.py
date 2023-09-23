















plt.plot(list(saves.keys()), [v['test_loss'] for v in saves.values()])
plt.ylabel('test_loss')
plt.xlabel('step')
plt.plot(list(saves.keys()), [v['train_loss'] for v in saves.values()])
plt.ylabel('train_loss')
plt.xlabel('step')
plt.plot(list(saves.keys()), [v['test_accuracy'] for v in saves.values()])
plt.ylabel('test_accuracy')
plt.xlabel('step')





























