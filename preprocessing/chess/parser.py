import re

def clean_chess_notation(text):
    text = re.sub(r'\s*\{[^}]*\}', '', text)  # Remove all stuff in {}
    text = text.replace('...', '.')  # Replace '...' with '.'
    text = re.sub(r'\.\s', '.', text)  # Remove whitespace after <number>.
    text = re.sub(r'\s(0-1|1-0|1/2-1/2)\s*$', '', text)  # Remove game end results from the end of the line
    text = re.sub(r'\?|\!', '', text)  # Remove chess annotations "?", "!", "!!" and "??"
    return text+'\n'

def writeline(line):
  if line.startswith("1/2"):
        f = open("draws.txt",'a')
        f.write(line)
        f.close()
  if "#" in line:
       f = open("wins.txt",'a')
       f.write(line)
       f.close()




def parse_chess_file(filename):
    with open(filename, 'r') as in_file:
            record = {}
            content = []
            parsing_content = False

            for line in in_file:
                if line.strip() == '':
                    if parsing_content:  # End of record
                        result_string = ' '.join([record['Result'], record['WhiteElo'], record['BlackElo']])
                        content_string = ' '.join(content)
                        writeline(clean_chess_notation(result_string + ' ' + content_string + '\n'))
                        record = {}
                        content = []
                        parsing_content = False
                    continue

                if line.startswith('['):
                    if parsing_content:  # If we're currently parsing content, then this is a new record
                        result_string = ' '.join([record['Result'], record['WhiteElo'], record['BlackElo']])
                        content_string = ' '.join(content)
                        out_file.write(result_string + ' ' + content_string + '\n')
                        record = {}
                        content = []
                    key, value = line.strip()[1:-1].split(' ', 1)  # Remove brackets, then split on the first space
                    record[key] = value.strip('"')  # Remove quotation marks
                else:
                    content.append(line.strip())
                    parsing_content = True

            # Write the last record to file, if any
            if record and content:
                result_string = ' '.join([record['Result'], record['WhiteElo'], record['BlackElo']])
                content_string = ' '.join(content)
                writeline(clean_chess_notation(result_string + ' ' + content_string + '\n'))


#parse_chess_file('sample.txt')
