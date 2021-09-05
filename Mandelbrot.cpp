#include <fstream>
#include <sstream>
#include "fracGen.hpp"
#include "rapidxml/rapidxml.hpp"

//Defines the default window width, height and the default maximum iterations
#define DEFAULT_WINDOW_WIDTH 1280
#define DEFAULT_WINDOW_HEIGHT 720
#define DEFAULT_MAX_ITERS 1024

void loadSettings(int& windowWidth, int& windowHeight, int& maxIters);

int main(int argc, char* argv[]) {
    int windowWidth, windowHeight, maxIters;

    loadSettings(windowWidth, windowHeight, maxIters);
    
    fracGen surveyor(windowWidth, windowHeight, maxIters);
    surveyor.start();

    return 0;
}

void loadSettings(int& windowWidth, int& windowHeight, int& maxIters)
{
    std::ifstream settingsFile("settings.xml");
    char* dataCString = nullptr;
    bool error = false;
    if (settingsFile.is_open())
    {
        char* temp = nullptr;
        std::string dataString((std::istreambuf_iterator<char>(settingsFile)), std::istreambuf_iterator<char>());
        settingsFile.close();
        dataCString = new char[dataString.length() + 1];
        strncpy(dataCString, dataString.c_str(), dataString.length());
        dataCString[dataString.length()] = '\0';

        rapidxml::xml_document<> settingsDoc;
        try
        {
            settingsDoc.parse<rapidxml::parse_non_destructive>(dataCString);
        }
        catch (rapidxml::parse_error)
        {
            error = true;
        }
        if (!error)
        {
            rapidxml::xml_node<>* settingsNode = settingsDoc.first_node("settings");

            temp = settingsNode->first_node("windowWidth")->value();
            if (temp == nullptr)
                error = true;
            else
                windowWidth = atoi(temp);

            temp = settingsNode->first_node("windowHeight")->value();
            if (temp == nullptr)
                error = true;
            else
                windowHeight = atoi(temp);

            temp = settingsNode->first_node("maxIters")->value();
            if (temp == nullptr)
                error = true;
            else
                maxIters = atoi(temp);
        }
    }
    else
        error = true;
    if (error)
    {
        std::cerr << "Error loading or parsing settings.xml. Setting to default values\n";
        windowWidth = DEFAULT_WINDOW_WIDTH;
        windowHeight = DEFAULT_WINDOW_HEIGHT;
        maxIters = DEFAULT_MAX_ITERS;
    }
    delete[] dataCString;
}