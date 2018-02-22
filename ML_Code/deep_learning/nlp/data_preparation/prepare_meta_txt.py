# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# split into words by white space
words = text.split()

# better to remove ponctuations, What's become Whats instead of 2 words What and s
re_punc = re.compile( ' [%s] ' % re.escape(string.punctuation))
stripped = [re_punc.sub( '' , w) for w in words]

# Example of removing non-printable characters.
re_print = re.compile( ' [^%s] ' % re.escape(string.printable))
result = [re_print.sub( '' , w) for w in words]