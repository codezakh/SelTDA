import cli
import functools

def test_parse_args(request, monkeypatch):
    with monkeypatch.context() as m:
        # Prevent the argparser from actually parsing the pytest args
        # and choking. We just override the parse_args method to always
        # parse the empty list.
        parse_args = cli.ArgumentParser.parse_args
        parse_empty_list = functools.partialmethod(parse_args, args=[])
        m.setattr(cli.ArgumentParser, 'parse_args', parse_empty_list)
        cli.parse_args(default_config_path='./configs/vqa.yaml')

def test_cli_setup(request, tmp_path_factory, monkeypatch):
    with monkeypatch.context() as m:
        # Prevent the argparser from actually parsing the pytest args
        # and choking. We just override the parse_args method to always
        # parse the empty list.
        parse_args = cli.ArgumentParser.parse_args
        parse_empty_list = functools.partialmethod(parse_args, args=[])
        m.setattr(cli.ArgumentParser, 'parse_args', parse_empty_list)
        args, config = cli.parse_args(default_config_path='./configs/vqa.yaml')
        args.output_dir = str(tmp_path_factory.mktemp('test_cli_setup'))
        cli.setup(args, config)