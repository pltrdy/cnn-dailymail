import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

tok_dir = "tok"
finished_files_dir = "finished_files"

VOCAB_SIZE = 200000

def to_bytes(_str):
  return _str.encode('utf-8')

def tokenize_stories(stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
  stories = os.listdir(stories_dir)
  # make IO list file
  print("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in stories:
      t = "%s.tok" % s
      f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, t)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
  subprocess.call(command)
  print("Stanford CoreNLP Tokenizer has finished.")
  os.remove("mapping.txt")

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  if num_orig != num_tokenized:
    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines

def fix_missing_period(line):
  """For compatibility only
  """
  return line


def get_art_abs(src_path, tgt_path):
  src = read_text_file(src_path)
  tgt = read_text_file(tgt_path)
  # Lowercase everything
  src = [line.lower() for line in src if line != ""]
  tgt = [line.lower() for line in tgt if line != ""]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  #lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  """
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)
  """
  # Make article into a single string
  article = ' '.join(src)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in tgt])

  return article, abstract


def write_to_bin(tok_files, out_file, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""

  num_stories = len(tok_files)

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(tok_files):
      if idx % 1000 == 0:
        print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      path = os.path.join(tok_dir, s)
      src_path = "%s.src.tok" % path
      tgt_path = "%s.tgt.tok" % path
      for _ in [src_path, tgt_path]:
        if not os.path.isfile(_):
          raise Exception("Error: Couldn't find tokenized file %s" % _)

      # Get the strings to write to .bin file
      article, abstract = [to_bytes(_) for _ in get_art_abs(src_path, tgt_path)]

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(b' ')
        abs_tokens = abstract.split(b' ')
        art_tokens = [t for t in art_tokens if t not in [to_bytes(SENTENCE_START), to_bytes(SENTENCE_END)]] # remove these tags from vocab
        abs_tokens = [t for t in abs_tokens if t not in [to_bytes(SENTENCE_START), to_bytes(SENTENCE_END)]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'wb') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + b' ' + to_bytes(str(count)) + b'\n')
    print("Finished writing vocab file")

"""
def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))
"""

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("USAGE: python make_datafiles.py <data_dir>")
    sys.exit()
  data_dir = sys.argv[1]

  # Check the stories directories contain the correct number of .story files
  #check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
  #check_num_stories(dm_stories_dir, num_expected_dm_stories)

  # Create some new directories
  if not os.path.exists(tok_dir): os.makedirs(tok_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  tokenize_stories(data_dir, tok_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(['train'], os.path.join(finished_files_dir, "train.bin"), makevocab=True)
  write_to_bin(['valid'], os.path.join(finished_files_dir, "valid.bin"))
  write_to_bin(['test'], os.path.join(finished_files_dir, "test.bin"))

  #write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
  #write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"), makevocab=True)
