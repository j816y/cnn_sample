#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>

#define IDEBUG  false
#define ODEBUG  false
#define PDEBUG  false

using namespace std;

//input size
int padding = 0;
int inputWidth = 6 + padding*2;
int inputDepth = 1;

//filter size
int numOfFilter = 2;
int filterRF = 3;
int filterDepth = inputDepth;
int stride = 1;

//https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
//for equation to calculate output width/height
int outputWidth = (inputWidth - filterRF + 2*padding)/stride + 1;
int outputDepth = numOfFilter;

//pool size
int poolSize = 2;   //2x2 pooling
int poolStride = 2; //non-overlapping pool
int poolOutputSize = (outputWidth - poolSize)/poolStride + 1;

//store in format inputVector[depth][height][width]
vector<vector<vector<double> > > inputVector;
//store in format filter[# of filters][depth][height][width]
vector<vector<vector<vector<double> > > > filter;
//store in format outputVector[depth][height][width]
vector<vector<vector<double> > > outputVector;
//store in format poolOutput[depth][height][width]
vector<vector<vector<double> > > poolOutput;


int init(){
    inputVector.resize(inputDepth);
    for(int i = 0; i < inputDepth; i++){
        inputVector[i].resize(inputWidth);
        for(int j = 0; j < inputWidth; j++){
            for (int k = 0; k < inputWidth; k++)
                inputVector[i][j].push_back(1);
        }
    }
    
    filter.resize(numOfFilter);
    for(int i = 0; i < numOfFilter; i++){
        filter[i].resize(filterDepth);
        for(int j = 0; j < filterDepth; j++){
            filter[i][j].resize(filterRF);
            for(int k = 0; k < filterRF; k++){
                for (int l = 0; l < filterRF; l++)
                    filter[i][j][k].push_back(i+10);
            }
        }
    }
    
    outputVector.resize(outputDepth);
    for(int i = 0; i < outputDepth; i++){
        outputVector[i].resize(outputWidth);
        for(int j = 0; j < outputWidth; j++){
            for (int k = 0; k < outputWidth; k++)
                outputVector[i][j].push_back(0);
        }
    }
    
    if (IDEBUG){
        cout << "Input: " << endl;
        for(int i = 0; i < inputDepth; i++){
            for(int j = 0; j < inputWidth; j++){
                for (int k = 0; k < inputWidth; k++)
                    cout << inputVector[i][j][k] << " ";
                cout << endl;
            }
        }
        getchar();
        
        cout << "Output: " << endl;
        for(int i = 0; i < outputDepth; i++){
            cout << "Depth " << i << endl;
            for(int j = 0; j < outputWidth; j++){
                for (int k = 0; k < outputWidth; k++)
                    cout << outputVector[i][j][k] << " ";
                cout << endl;
            }
        }
        getchar();
        
        for(int i = 0; i < filter.size(); i++){
            cout << "Filter " << i << endl;
            for(int j = 0; j < filter[i].size(); j++){
                for(int k = 0; k < filter[i][j].size(); k++){
                    for (int l = 0; l < filter[i][j][k].size();l++)
                        cout << filter[i][j][k][l] << " ";
                    cout << endl;
                }
                cout << endl;
            }
        }
        getchar();
    }
    return 0;
}

int conv(){
    for (int f = 0; f < outputDepth; f++){                      //outputDepth = # of neurons = # of filters
        for (int v = 0; v < outputWidth; v++){                  //v for vertical (outputHeight)
            for (int h = 0; h < outputWidth; h++){              //h for horizontal (outputWidth)
                for (int y = 0; y < filterRF; y++){             //filter height = width
                    for (int x = 0; x < filterRF; x++){         //filter width
                        for (int z = 0; z < filterDepth; z++){  //filter depth
                            outputVector[f][v][h] += filter[f][z][y][x] * inputVector[z][v+y][h+x];
                        }
                    }
                }
            }
        }
    }
    
    if(ODEBUG){
        for (int i = 0; i < outputVector.size(); i++){
            cout << "Depth: " << i << endl;
            for (int j = 0; j < outputVector[i].size(); j++){
                for (int k = 0; k < outputVector[k].size(); k++){
                    cout << outputVector[i][j][k] << " " << endl;
                }
                cout << endl;
            }
            cout << endl;
        }
        getchar();
    }
    return 0;
}

int maxpool(){
    int value = 0;
    for (int i = 0; i < outputVector.size(); i++){      //outputdepth
        for (int j = 0; j < poolOutputSize; j++){       //vertical
            for (int k = 0; k < poolOutputSize; k++){   //horizontal
                for (int y = 0; y < poolSize; y++){     //vertical
                    for (int x = 0; x < poolSize; x++){ //horizontal
                        value = outputVector[i][j*poolStride+y][k*poolStride+x];
                        if (value > poolOutput[i][j][k])
                            poolOutput[i][j][k] = value;
                    }
                }
            }
        }
    }
    
    if(PDEBUG){
        for(int i = 0; i < poolOutput.size();i++){
            cout << "Depth: " << i << endl;
            for (int j = 0; j < poolOutput[i].size(); j++){
                for (int k = 0; k < poolOutput[k].size(); k++){
                    cout << poolOutput[i][j][k] << " " << endl;
                }
                cout << endl;
            }
        }
        getchar();
    }
    return 0;
}



int main(){
    int error = 0;
    error = init();
    error = conv();
    error = maxpool();
    return 0;
}