import struct
import sys

def read_gguf_metadata(filepath):
    with open(filepath, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'GGUF':
            print('Not a GGUF file')
            return

        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

        print(f'GGUF version: {version}')
        print(f'Metadata entries: {metadata_kv_count}')
        print('')

        # Read metadata key-value pairs
        for i in range(metadata_kv_count):
            # Read key length and key
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')

            # Read value type
            value_type = struct.unpack('<I', f.read(4))[0]

            # Read value based on type
            if value_type == 8:  # STRING
                str_len = struct.unpack('<Q', f.read(8))[0]
                try:
                    value = f.read(str_len).decode('utf-8')
                    if 'name' in key.lower() or 'model' in key.lower() or 'arch' in key.lower():
                        print(f'{key}: {value}')
                except:
                    # Skip if can't decode
                    pass
            else:
                # Skip non-string values for now
                pass

import sys
filepath = sys.argv[1] if len(sys.argv) > 1 else 'gemma-3-1b-it-f16.gguf'
read_gguf_metadata(filepath)
