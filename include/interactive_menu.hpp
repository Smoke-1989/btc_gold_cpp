#ifndef INTERACTIVE_MENU_HPP
#define INTERACTIVE_MENU_HPP

#include <string>
#include <vector>
#include <map>
#include <cmath>

struct BTCGoldConfig {
    int threads;
    std::string input_type;
    std::string input_file;
    int search_mode;
    bool stop_on_find;
    bool verbose;
    
    // Mode-specific string parameters
    std::map<std::string, std::string> mode_params;
    
    // Mode-specific integer parameters
    std::map<std::string, int> mode_params_int;
};

class InteractiveMenu {
public:
    InteractiveMenu();
    
    // Main flow
    void run();
    
    // UI methods
    void print_banner();
    void print_main_menu();
    void show_modes();
    void show_mode_details(int mode);
    void configure_parameters();
    void configure_mode_parameters();
    void view_configuration();
    void show_help();
    
    // Utility
    std::string get_mode_name(int mode);
    BTCGoldConfig get_configuration();
    
private:
    BTCGoldConfig config_;
};

#endif // INTERACTIVE_MENU_HPP
