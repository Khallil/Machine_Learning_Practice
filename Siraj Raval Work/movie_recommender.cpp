#include "commands.h"
#include <iostream>

int main() {
    retrieveDataset("https://s3-us-west-2.amazonaws.com/amazon-dsstne-samples/data/ml20m-all");
    generateInputLayer("ml20m-all");
    generateOutputLayer("ml20m-all");
    train(256,10);
    predict(10, "ml20m-all ")
    return 0;
}

/* Le code est créé maintenant il faut le run
sur un serveur amazon ubuntu option GPU sauf que 
je trouve pas d'AMI pour installer le code dessus */