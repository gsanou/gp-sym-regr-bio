#!/usr/bin/env perl

use strict;
use warnings;
use Cwd qw(cwd);
use File::Basename qw(fileparse);
use File::Find;
use File::Spec;
use Getopt::Long qw(:config auto_help auto_version);
use IO::Uncompress::Gunzip qw(gunzip $GunzipError);
use Pod::Usage qw(pod2usage);
use Data::Dumper;

$Data::Dumper::Sortkeys = 1;
$Data::Dumper::Terse = 1;
$Data::Dumper::Deepcopy = 1;

# unbuffer error and output streams (make sure STDOUT
# is last so that it remains the default filehandle)
select(STDERR); $| = 1;
select(STDOUT); $| = 1;

my @file_base_exts = qw(
    htseq.counts
    star.counts
    FPKM
    FPKM-UQ
);
my @file_add_exts = qw(
    txt
);
my @compr_exts = qw(
    gz
);
my @sample_sheet_read_cols = (
    'File Name',
    'Case ID',
    'Sample ID',
);
my $sample_sheet_glob = 'gdc_sample_sheet.*.tsv';
my %star_stranded_col_idx = (
    'no' => 0,
    'yes' => 1,
    'reverse' => 2,
);

my $data_dir = cwd();
my $sample_sheet = '';
my $star_stranded = 'no';
my $verbose = 0;
GetOptions(
    'data-dir:s' => \$data_dir,
    'sample-sheet:s' => \$sample_sheet,
    'star-stranded:s' => \$star_stranded,
    'verbose' => \$verbose,
) || pod2usage(-verbose => 0);
if (!-d $data_dir) {
    pod2usage(-message => "Invalid --data-dir: $data_dir");
}
if ($star_stranded ne 'no' and $star_stranded ne 'yes' and $star_stranded ne 'reverse') {
    pod2usage(-message => "Invalid --star-stranded: must be no, yes or reverse");
}
my @data_dir_paths = File::Spec->splitdir($data_dir);
my $data_dir_name = $data_dir_paths[$#data_dir_paths];
my $sample_sheet_exists = 0;
if ($sample_sheet) {
    $sample_sheet_exists++;
}
else {
    $sample_sheet = <$data_dir/$sample_sheet_glob>;
    $sample_sheet_exists++ if $sample_sheet;
}
my %uuid2sample_id;
if ($sample_sheet_exists) {
    if (!-f $sample_sheet) {
        pod2usage(-message => "Invalid --sample-sheet: $sample_sheet");
    }
    open(my $in_fh, '<', $sample_sheet) or die "$sample_sheet: $!";
    chomp(my $col_header_line = <$in_fh>);
    my @col_headers = split(/\t/, $col_header_line);
    my %col_header_idx = map { $col_headers[$_] => $_ } 0 .. $#col_headers;
    for my $read_col (@sample_sheet_read_cols) {
        if (not defined $col_header_idx{$read_col}) {
            die "ERROR: $read_col missing from sample sheet\n";
        }
    }
    while (my $line = <$in_fh>) {
        chomp($line);
        my @fields = split(/\t/, $line);
        my $uuid = fileparse($fields[$col_header_idx{'File Name'}], qr/\..*/);
        $uuid2sample_id{$uuid} = $fields[$col_header_idx{'Sample ID'}];
    }
    close($in_fh);
}
my $file_add_exts_pattern = join('|', @file_add_exts);
my $file_add_exts_regex = qr/$file_add_exts_pattern/o;
my $compr_exts_pattern = join('|', @compr_exts);
my $compr_exts_regex = qr/$compr_exts_pattern/o;
for my $file_base_ext (@file_base_exts) {
    my (@data_matrix, @sample_ids, @feature_ids);
    my $sample_idx = 0;
    my $file_count = 0;
    find({
        wanted => sub {
            return unless -f and m/
                \.$file_base_ext
                (?:\.(?:$file_add_exts_regex))?
                (?:\.($compr_exts_regex))?$
            /ix;
            my $file = $_;
            my $compr_ext = $1;
            if ($verbose) {
                print "Processing $file_base_ext files\n" if $file_count == 0;
                print "$file\n";
                $file_count++;
            }
            else {
                print "\rProcessing ", ++$file_count, " $file_base_ext files";
            }
            my $file_id = fileparse($file, qr/\..*/);
            if ($sample_sheet_exists) {
                if (defined $uuid2sample_id{$file_id}) {
                    $sample_ids[$sample_idx] = $uuid2sample_id{$file_id};
                }
                else {
                    print "\r" unless $verbose;
                    warn "ERROR: $file not found in sample sheet\n";
                    return;
                }
            }
            else {
                $sample_ids[$sample_idx] = $file_id;
            }
            my $feature_idx = 0;
            my $file_str;
            if (defined($compr_ext)) {
                if (lc($compr_ext) eq 'gz') {
                    gunzip $file => \$file_str
                        or die "$file gunzip failed: $GunzipError";
                }
            }
            open(my $in_fh, '<', defined($file_str) ? \$file_str : $file)
                or die "$file: $!";
            while (my $line = <$in_fh>) {
                chomp($line);
                my ($feature_id, @values) = split(/\t/, $line);
                if (!@feature_ids or $feature_idx > $#feature_ids) {
                    $feature_ids[$feature_idx] = $feature_id;
                }
                elsif ($feature_ids[$feature_idx] ne $feature_id) {
                    die "ERROR: $feature_id not in same order\n";
                }
                $data_matrix[$sample_idx][$feature_idx] =
                    $file_base_ext eq 'star.counts'
                        ? $values[$star_stranded_col_idx{$star_stranded}]
                        : $values[0];
                $feature_idx++;
            }
            close($in_fh);
            $sample_idx++;
        }
    }, $data_dir);
    if (@data_matrix) {
        print "\n" unless $verbose;
        my @sorted_sample_id_idxs = sort {
            $sample_ids[$a] cmp $sample_ids[$b]
        } 0 .. $#sample_ids;
        my @sorted_sample_ids = @sample_ids[@sorted_sample_id_idxs];
        my @sorted_data_matrix = @data_matrix[@sorted_sample_id_idxs];
        my $sorted_data_matrix_ref = transpose_matrix(\@sorted_data_matrix);
        my $out_file = "$data_dir_name.\L$file_base_ext\E.tsv";
        print "Writing $out_file\n";
        open(my $out_fh, '>', $out_file) or die "$out_file: $!";
        print $out_fh "ID_REF\t", join("\t", @sorted_sample_ids), "\n";
        for my $feature_idx (0 .. $#{$sorted_data_matrix_ref}) {
            print $out_fh "$feature_ids[$feature_idx]\t",
                join("\t", @{$sorted_data_matrix_ref->[$feature_idx]}), "\n";
        }
        close($out_fh);
    }
}

sub transpose_matrix {
    my ($mat_ref) = @_;
    my $mat_tr_ref;
    for my $row (0 .. $#{$mat_ref}) {
        for my $col (0 .. $#{$mat_ref->[$row]}) {
            $mat_tr_ref->[$col]->[$row] = $mat_ref->[$row]->[$col];
        }
    }
    return $mat_tr_ref;
}

__END__

=head1 NAME

create_data_matrix.pl - Create a data matrix from a GDC cart download or other individual files

=head1 SYNOPSIS

 create_data_matrix.pl [options]

 Options:
    --data-dir <dir>        Data directory
                            (default = current directory)
    --sample-sheet <file>   GDC sample sheet
                            (optional)
    --star-stranded <str>   STAR counts stranded (no, yes, reverse)
                            (default = no)
    --verbose               Be verbose
    --help                  Display usage and exit
    --version               Display program version and exit

=cut
