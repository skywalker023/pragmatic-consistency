#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import parlai.core.params as params
import parlai.core.build_data as build_data


FOLDER_NAME = 'self_conscious_dialogue'


def build(opt):
    dpath = os.path.join(opt['datapath'], FOLDER_NAME)
    # version 1.0: initial release
    version = '1.0'

    # check whether data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')

        # make a clean directory if needed
        if build_data.built(dpath):
            # if an older version exists, remove those outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        #########################
        # ConvAI2 (PersonaChat)
        #########################
        fname = 'data_v1.tar.gz'
        url = 'https://parl.ai/downloads/controllable_dialogue/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)

        fname = 'convai2_fix_723.tgz'
        url = 'http://parl.ai/downloads/convai2/' + fname
        build_data.download(url, dpath, fname)
        build_data.untar(dpath, fname)

        #########################
        # Dialogue NLI
        #########################
        fname = 'dialogue_nli.zip'
        gd_id = '1WtbXCv3vPB5ql6w0FVDmAEMmWadbrCuG'
        build_data.download_from_google_drive(gd_id, os.path.join(dpath, fname))
        build_data.untar(dpath, fname)

        fname = 'dialogue_nli_evaluation.zip'
        gd_id = '1sllq30KMJzEVQ4C0-a9ShSLSPIZc3iMi'
        build_data.download_from_google_drive(gd_id, os.path.join(dpath, fname))
        build_data.untar(dpath, fname)

        #########################
        # Distractor personas
        #########################
        fname = 'train_sorted_50_personas.json'
        gd_id = '1SGFdJqyNYeepKFqwMLv4Ym717QQTtpi8'
        build_data.download_from_google_drive(gd_id, os.path.join(dpath, fname))
        fname = 'valid_sorted_50_personas.json'
        gd_id = '1A7oVKmjJ1EZTh6-3Gio4XQo81QgnTGGi'
        build_data.download_from_google_drive(gd_id, os.path.join(dpath, fname))
        fname = 'dnli_sorted_50_personas.json'
        gd_id = '1wlIkVcBZoGQd3rbI7XWNhuq4rvw9FyoP'
        build_data.download_from_google_drive(gd_id, os.path.join(dpath, fname))

        print("Data has been placed in " + dpath)

        build_data.mark_done(dpath, version)


def make_path(opt, fname):
    return os.path.join(opt['datapath'], FOLDER_NAME, fname)


if __name__ == '__main__':
    opt = params.ParlaiParser().parse_args(print_args=False)
    build(opt)
