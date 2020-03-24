# -*- coding: utf-8 -*-
import json


class JsonParse(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.__load_json__()
        self.__parse_json__()

    def __load_json__(self):
        with open(self.file_path, 'r') as f:
            js = json.load(f)
        assert len(js) > 0, 'Json file is empty'
        self.js = js

    def __parse_json__(self):
        raise NotImplementedError

    def __repr__(self):
        format_string = 'Namespace{}'.format(self.js)
        return format_string

    def format_str(self, string):
        return string.lower().replace('-', '_').replace(' ', '')


class ArgsParse(JsonParse):
    def __init__(self, file_path):
        super(ArgsParse, self).__init__(file_path)

    def __parse_json__(self):
        for k, v in self.js.items():
            k = k.replace('-', '_')
            if not isinstance(v, dict):
                setattr(self, k, v)
            else:
                assert 'value' in v.keys(), \
                    "Need `value` in second level attributes."
                if v['value'] is not None:
                    setattr(self, k, v['value'])
                    continue
                assert 'default' in v.keys(), \
                    'If value is `None` need a default value.'
                setattr(self, k, v['default'])


if __name__ == '__main__':
    A = ArgsParse('./train_config.json')
    print(A.devices)
