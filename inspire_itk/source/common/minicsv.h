
#ifndef MINI_CSV_H
#define MINI_CSV_H

#include <iostream>
#include <stdio.h>
#include <string>

namespace itk {
    struct CSVFile {
        std::vector<std::vector<std::string> > contents;
        bool success;
    };

    int ReadString(FILE* f, bool is_first, std::string& s) {
        char c;
        while(fread(&c, 1, 1, f) == 1U) {
            if(c == ',') {
                return 0;
            } else if((c == '\n' || c == '\r')) {
                if(s.length() > 0 || (!is_first)) {
                    return 1;
                } else {
                    // Skip
                }
            } else {
				s.push_back(c);
            }
        }
        return -1; // eof
    }
    int ReadRow(FILE* f, std::vector<std::string>& row) {
        while(true) {
            std::string s;
            int r = ReadString(f, row.size() == 0, s);
            if(r == 0) { // end of string
                row.push_back(s);
            } else if(r == 1) { // end of row
                row.push_back(s);
                return 0;
            } else { // end of file
                if(row.size() > 0 || s.length() > 0) {
                    row.push_back(s);
                }
                return -1;
            }
        }
    }
    CSVFile ReadCSV(const char* path) {
        CSVFile csv;

        FILE *f = fopen(path, "rb");

        if(!f) {
            csv.success = false;
            return csv;
        }

        bool go = true;

        while(go) {
            std::vector<std::string> row;

            int r = ReadRow(f, row);
            if(r == -1) {
                go = false;
            }
            if(row.size() > 0) {
                csv.contents.push_back(row);
            }
        }

        csv.success = true;
        return csv;
    }
}

#endif
